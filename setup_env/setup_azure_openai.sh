#!/usr/bin/env bash

# Create the Azure resources required to generate answers and search embeddings,
# then write the resulting Azure OpenAI and Azure AI Search configuration to .env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "setup_env" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

LOCATION="norwayeast"
RESOURCE_PREFIX=""
ENV_FILE="${PROJECT_ROOT}/.env"
EMBEDDING_MODEL_NAME="text-embedding-3-small"
EMBEDDING_MODEL_VERSION="1"
EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-small"
EMBEDDING_DEPLOYMENT_SKU="GlobalStandard"
EMBEDDING_DEPLOYMENT_CAPACITY="10"
CHAT_MODEL_NAME="gpt-5-mini"
CHAT_MODEL_VERSION="2025-08-07"
CHAT_DEPLOYMENT_NAME="gpt-5-mini"
CHAT_DEPLOYMENT_SKU="GlobalStandard"
CHAT_DEPLOYMENT_CAPACITY="10"
SEARCH_INDEX_NAME="insurance-policy-chunks"
VECTOR_ALGORITHM="hnsw"
TEST_OPENAI_SCRIPT="${PROJECT_ROOT}/src/setup_env/test_openai_service.py"
INDEX_CHUNKS_SCRIPT="${PROJECT_ROOT}/src/setup_env/index_document_chunks.py"
IP_CHUNKS_FILE="${PROJECT_ROOT}/data/processed/ip_document_chunks.jsonl"
HH_CHUNKS_FILE="${PROJECT_ROOT}/data/processed/hh_document_chunks.jsonl"

usage() {
    cat <<EOF
Usage: $0 --prefix NAME [options]

Required:
  --prefix NAME          Prefix used to name the Azure resources

Options:
  --location LOCATION    Azure region for the resources (default: $LOCATION)
  --env-file PATH        Environment file to update (default: $ENV_FILE)
  --capacity NUMBER      Embedding deployment capacity in thousands of tokens
                         per minute (default: $EMBEDDING_DEPLOYMENT_CAPACITY)
  --chat-model NAME      Chat model to deploy (default: $CHAT_MODEL_NAME)
  --chat-model-version V Chat model version (default: $CHAT_MODEL_VERSION)
  --chat-deployment NAME Chat deployment name (default: $CHAT_DEPLOYMENT_NAME)
  --chat-capacity NUMBER Chat deployment capacity in thousands of tokens per
                         minute (default: $CHAT_DEPLOYMENT_CAPACITY)
  --vector-algorithm ALG  Vector index algorithm: hnsw or eknn
                         (default: $VECTOR_ALGORITHM)
  -h, --help             Show this help message
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

info() {
    echo ">>> $*"
}

require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "'$command_name' is not installed. Install it before running this script."
}

provisioning_succeeded() {
    [[ "$1" =~ ^[Ss][Uu][Cc][Cc][Ee][Ee][Dd][Ee][Dd]$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || fail "--prefix requires a value."
            RESOURCE_PREFIX="$2"
            shift 2
            ;;
        --location)
            [[ $# -ge 2 ]] || fail "--location requires a value."
            LOCATION="$2"
            shift 2
            ;;
        --env-file)
            [[ $# -ge 2 ]] || fail "--env-file requires a value."
            ENV_FILE="$2"
            shift 2
            ;;
        --capacity)
            [[ $# -ge 2 ]] || fail "--capacity requires a value."
            EMBEDDING_DEPLOYMENT_CAPACITY="$2"
            shift 2
            ;;
        --chat-model)
            [[ $# -ge 2 ]] || fail "--chat-model requires a value."
            CHAT_MODEL_NAME="$2"
            shift 2
            ;;
        --chat-model-version)
            [[ $# -ge 2 ]] || fail "--chat-model-version requires a value."
            CHAT_MODEL_VERSION="$2"
            shift 2
            ;;
        --chat-deployment)
            [[ $# -ge 2 ]] || fail "--chat-deployment requires a value."
            CHAT_DEPLOYMENT_NAME="$2"
            shift 2
            ;;
        --chat-capacity)
            [[ $# -ge 2 ]] || fail "--chat-capacity requires a value."
            CHAT_DEPLOYMENT_CAPACITY="$2"
            shift 2
            ;;
        --vector-algorithm)
            [[ $# -ge 2 ]] || fail "--vector-algorithm requires a value."
            VECTOR_ALGORITHM="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown option: $1"
            ;;
    esac
done

[[ -n "$RESOURCE_PREFIX" ]] || fail "--prefix is required."
if [[ "$ENV_FILE" != /* ]]; then
    ENV_FILE="${PROJECT_ROOT}/${ENV_FILE#./}"
fi
[[ "$RESOURCE_PREFIX" =~ ^[a-zA-Z0-9][a-zA-Z0-9-]*$ ]] ||
    fail "--prefix must start with a letter or number and contain only letters, numbers, and hyphens."
[[ "$EMBEDDING_DEPLOYMENT_CAPACITY" =~ ^[1-9][0-9]*$ ]] ||
    fail "--capacity must be a positive integer."
[[ "$CHAT_DEPLOYMENT_CAPACITY" =~ ^[1-9][0-9]*$ ]] ||
    fail "--chat-capacity must be a positive integer."
[[ -n "$CHAT_MODEL_NAME" ]] || fail "--chat-model cannot be empty."
[[ -n "$CHAT_MODEL_VERSION" ]] || fail "--chat-model-version cannot be empty."
[[ "$CHAT_DEPLOYMENT_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] ||
    fail "--chat-deployment contains unsupported characters."
[[ "$VECTOR_ALGORITHM" == "hnsw" || "$VECTOR_ALGORITHM" == "eknn" ]] ||
    fail "--vector-algorithm must be 'hnsw' or 'eknn'."

require_command az
require_command awk
require_command openssl
require_command python3

[[ -f "$TEST_OPENAI_SCRIPT" ]] ||
    fail "OpenAI test script not found: $TEST_OPENAI_SCRIPT"
[[ -f "$INDEX_CHUNKS_SCRIPT" ]] ||
    fail "Chunk indexing script not found: $INDEX_CHUNKS_SCRIPT"

if ! az account get-access-token --output none >/dev/null 2>&1; then
    fail "You are not logged into Azure, or your session has expired. Run 'az login' and try again."
fi

SUBSCRIPTION_ID="$(az account show --query id --output tsv)"
SUBSCRIPTION_NAME="$(az account show --query name --output tsv)"

# The Azure OpenAI account name must be globally unique. A stable suffix makes
# the script rerunnable for the same prefix and Azure subscription.
ACCOUNT_SUFFIX="$({ printf '%s' "${SUBSCRIPTION_ID}:${RESOURCE_PREFIX}"; } | openssl dgst -sha256 | awk '{print substr($NF, 1, 8)}')"
NORMALIZED_PREFIX="$(printf '%s' "$RESOURCE_PREFIX" | tr '[:upper:]' '[:lower:]')"
OPENAI_PREFIX="${NORMALIZED_PREFIX:0:47}"
SEARCH_PREFIX="${NORMALIZED_PREFIX:0:43}"

RESOURCE_GROUP="${RESOURCE_PREFIX}-rg"
AZURE_OPENAI_ACCOUNT="${OPENAI_PREFIX}-openai-${ACCOUNT_SUFFIX}"
SEARCH_SERVICE="${SEARCH_PREFIX}-search-${ACCOUNT_SUFFIX}"

info "Signed in to subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"

create_resource_group() {
    local provisioning_state

    info "Creating or updating resource group '$RESOURCE_GROUP' in '$LOCATION'..."
    provisioning_state="$(
        az group create \
            --name "$RESOURCE_GROUP" \
            --location "$LOCATION" \
            --tags "project=$RESOURCE_PREFIX" \
            --query properties.provisioningState \
            --output tsv
    )"

    provisioning_succeeded "$provisioning_state" ||
        fail "Resource group provisioning state: $provisioning_state"
}

create_azure_openai_account() {
    local provisioning_state

    if az cognitiveservices account show \
        --name "$AZURE_OPENAI_ACCOUNT" \
        --resource-group "$RESOURCE_GROUP" \
        --output none >/dev/null 2>&1; then
        info "Azure OpenAI account '$AZURE_OPENAI_ACCOUNT' already exists."
        return
    fi

    info "Creating Azure OpenAI account '$AZURE_OPENAI_ACCOUNT'..."
    provisioning_state="$(
        az cognitiveservices account create \
            --name "$AZURE_OPENAI_ACCOUNT" \
            --resource-group "$RESOURCE_GROUP" \
            --location "$LOCATION" \
            --kind OpenAI \
            --sku S0 \
            --custom-domain "$AZURE_OPENAI_ACCOUNT" \
            --yes \
            --query properties.provisioningState \
            --output tsv
    )"

    provisioning_succeeded "$provisioning_state" ||
        fail "Azure OpenAI account provisioning state: $provisioning_state"
}

create_embedding_deployment() {
    local provisioning_state

    info "Creating or updating Global Standard embedding deployment '$EMBEDDING_DEPLOYMENT_NAME'..."
    az cognitiveservices account deployment create \
        --name "$AZURE_OPENAI_ACCOUNT" \
        --resource-group "$RESOURCE_GROUP" \
        --deployment-name "$EMBEDDING_DEPLOYMENT_NAME" \
        --model-format OpenAI \
        --model-name "$EMBEDDING_MODEL_NAME" \
        --model-version "$EMBEDDING_MODEL_VERSION" \
        --sku-name "$EMBEDDING_DEPLOYMENT_SKU" \
        --sku-capacity "$EMBEDDING_DEPLOYMENT_CAPACITY" \
        --only-show-errors \
        --output none

    provisioning_state="$(
        az cognitiveservices account deployment show \
            --name "$AZURE_OPENAI_ACCOUNT" \
            --resource-group "$RESOURCE_GROUP" \
            --deployment-name "$EMBEDDING_DEPLOYMENT_NAME" \
            --query properties.provisioningState \
            --output tsv
    )"

    provisioning_succeeded "$provisioning_state" ||
        fail "Embedding deployment provisioning state: $provisioning_state"
}

create_chat_deployment() {
    local provisioning_state

    info "Creating or updating Global Standard chat deployment '$CHAT_DEPLOYMENT_NAME'..."
    az cognitiveservices account deployment create \
        --name "$AZURE_OPENAI_ACCOUNT" \
        --resource-group "$RESOURCE_GROUP" \
        --deployment-name "$CHAT_DEPLOYMENT_NAME" \
        --model-format OpenAI \
        --model-name "$CHAT_MODEL_NAME" \
        --model-version "$CHAT_MODEL_VERSION" \
        --sku-name "$CHAT_DEPLOYMENT_SKU" \
        --sku-capacity "$CHAT_DEPLOYMENT_CAPACITY" \
        --only-show-errors \
        --output none

    provisioning_state="$(
        az cognitiveservices account deployment show \
            --name "$AZURE_OPENAI_ACCOUNT" \
            --resource-group "$RESOURCE_GROUP" \
            --deployment-name "$CHAT_DEPLOYMENT_NAME" \
            --query properties.provisioningState \
            --output tsv
    )"

    provisioning_succeeded "$provisioning_state" ||
        fail "Chat deployment provisioning state: $provisioning_state"
}

create_search_service() {
    local create_help
    local -a create_arguments
    local provisioning_state

    if az search service show \
        --name "$SEARCH_SERVICE" \
        --resource-group "$RESOURCE_GROUP" \
        --output none >/dev/null 2>&1; then
        info "Azure AI Search service '$SEARCH_SERVICE' already exists."
        return
    fi

    info "Creating Azure AI Search service '$SEARCH_SERVICE' on the Free tier..."
    create_arguments=(
        search service create
        --name "$SEARCH_SERVICE"
        --resource-group "$RESOURCE_GROUP"
        --location "$LOCATION"
        --sku free
    )

    create_help="$(az search service create --help 2>&1)"
    if [[ "$create_help" == *"--knowledge-retrieval"* ]]; then
        create_arguments+=(--knowledge-retrieval free)
        info "Enabling the Foundry IQ free knowledge-retrieval plan explicitly."
    else
        info "This Azure CLI version does not expose --knowledge-retrieval; continuing with the Free Search tier."
    fi

    provisioning_state="$(
        az "${create_arguments[@]}" \
            --query provisioningState \
            --output tsv
    )"

    provisioning_succeeded "$provisioning_state" ||
        fail "Azure AI Search service provisioning state: $provisioning_state"
}

update_env_value() {
    local key="$1"
    local value="$2"
    local env_directory
    local temp_file

    env_directory="$(dirname "$ENV_FILE")"
    mkdir -p "$env_directory"
    touch "$ENV_FILE"
    temp_file="$(mktemp "${env_directory}/.env.XXXXXX")"

    awk -v key="$key" -v value="$value" '
        BEGIN { updated = 0 }
        $0 ~ "^[[:space:]]*(export[[:space:]]+)?" key "=" {
            if (!updated) {
                print key "=" value
                updated = 1
            }
            next
        }
        { print }
        END {
            if (!updated) {
                print key "=" value
            }
        }
    ' "$ENV_FILE" > "$temp_file"

    chmod --reference="$ENV_FILE" "$temp_file" 2>/dev/null || chmod 600 "$temp_file"
    mv "$temp_file" "$ENV_FILE"
}

write_environment_file() {
    local endpoint
    local api_key
    local search_endpoint
    local search_admin_key

    info "Retrieving the Azure OpenAI endpoint and API key..."
    endpoint="$(
        az cognitiveservices account show \
            --name "$AZURE_OPENAI_ACCOUNT" \
            --resource-group "$RESOURCE_GROUP" \
            --query properties.endpoint \
            --output tsv
    )"
    api_key="$(
        az cognitiveservices account keys list \
            --name "$AZURE_OPENAI_ACCOUNT" \
            --resource-group "$RESOURCE_GROUP" \
            --query key1 \
            --output tsv
    )"

    [[ -n "$endpoint" ]] || fail "Azure OpenAI endpoint was empty."
    [[ -n "$api_key" ]] || fail "Azure OpenAI API key was empty."

    info "Retrieving the Azure AI Search endpoint and admin key..."
    search_endpoint="https://${SEARCH_SERVICE}.search.windows.net"
    if az search service admin-key list --help >/dev/null 2>&1; then
        search_admin_key="$(
            az search service admin-key list \
                --search-service-name "$SEARCH_SERVICE" \
                --resource-group "$RESOURCE_GROUP" \
                --query primaryKey \
                --output tsv
        )"
    else
        search_admin_key="$(
            az search admin-key show \
                --service-name "$SEARCH_SERVICE" \
                --resource-group "$RESOURCE_GROUP" \
                --query primaryKey \
                --output tsv
        )"
    fi

    [[ -n "$search_admin_key" ]] || fail "Azure AI Search admin key was empty."

    update_env_value "AZURE_OPENAI_ENDPOINT" "$endpoint"
    update_env_value "AZURE_OPENAI_API_KEY" "$api_key"
    update_env_value "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" "$EMBEDDING_DEPLOYMENT_NAME"
    update_env_value "AZURE_OPENAI_CHAT_DEPLOYMENT" "$CHAT_DEPLOYMENT_NAME"
    update_env_value "AZURE_SEARCH_ENDPOINT" "$search_endpoint"
    update_env_value "AZURE_SEARCH_ADMIN_KEY" "$search_admin_key"
    update_env_value "AZURE_SEARCH_INDEX_NAME" "$SEARCH_INDEX_NAME"

    info "Updated '$ENV_FILE' with the Azure OpenAI and Azure AI Search settings."
}

test_openai_service() {
    info "Testing the Azure OpenAI embedding deployment..."

    if ! (cd "$PROJECT_ROOT" && python3 "$TEST_OPENAI_SCRIPT"); then
        fail "Azure OpenAI service test failed. The document chunks were not indexed."
    fi

    info "Azure OpenAI service test completed successfully."
}

check_processed_chunks() {
    local chunk_file
    local chunk_count

    info "Checking the processed document chunks..."

    for chunk_file in "$IP_CHUNKS_FILE" "$HH_CHUNKS_FILE"; do
        [[ -f "$chunk_file" ]] || fail "Processed chunk file not found: $chunk_file"
        [[ -s "$chunk_file" ]] || fail "Processed chunk file is empty: $chunk_file"

        chunk_count="$(awk 'END { print NR }' "$chunk_file")"
        info "Found $chunk_count chunks in '$chunk_file'."
    done
}

index_document_chunks() {
    info "Creating the Azure AI Search index with '$VECTOR_ALGORITHM' and indexing the document chunks..."

    if ! (
        cd "$PROJECT_ROOT"
        python3 "$INDEX_CHUNKS_SCRIPT" \
            "$IP_CHUNKS_FILE" \
            "$HH_CHUNKS_FILE" \
            --vector-algorithm "$VECTOR_ALGORITHM"
    ); then
        fail "Document chunk indexing failed."
    fi

    info "Document chunks indexed successfully."
}

create_resource_group
create_azure_openai_account
create_embedding_deployment
create_chat_deployment
create_search_service
write_environment_file
test_openai_service
check_processed_chunks
index_document_chunks

info "Azure OpenAI and Azure AI Search setup completed successfully."
info "Resource group: $RESOURCE_GROUP"
info "Azure OpenAI account: $AZURE_OPENAI_ACCOUNT"
info "Embedding deployment: $EMBEDDING_DEPLOYMENT_NAME"
info "Chat deployment: $CHAT_DEPLOYMENT_NAME ($CHAT_MODEL_NAME, version $CHAT_MODEL_VERSION)"
info "Azure AI Search service: $SEARCH_SERVICE"
info "Search index: $SEARCH_INDEX_NAME"
info "Vector algorithm: $VECTOR_ALGORITHM"
