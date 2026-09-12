#!/usr/bin/env bash

set -euo pipefail

if (( $# != 3 )); then
    printf 'Usage: %s SOURCE_DIRECTORY DATASET_SLUG VERSION\n' "$0" >&2
    exit 2
fi

source_directory="${1%/}"
dataset_slug="$2"
version="$3"

if [[ ! -d "$source_directory" ]]; then
    printf 'Source directory does not exist: %s\n' "$source_directory" >&2
    exit 2
fi

if [[ ! "$dataset_slug" =~ ^[a-z0-9][a-z0-9-]*$ ]] || [[ ! "$version" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Dataset slug or version contains unsupported characters.\n' >&2
    exit 2
fi

for command_name in rclone shasum; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Required command is not installed: %s\n' "$command_name" >&2
        exit 1
    fi
done

: "${R2_BUCKET:?Set R2_BUCKET to the destination bucket name}"
: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID to your Cloudflare account ID}"
: "${R2_ACCESS_KEY_ID:?Set R2_ACCESS_KEY_ID to your R2 access key}"
: "${R2_SECRET_ACCESS_KEY:?Set R2_SECRET_ACCESS_KEY to your R2 secret key}"

endpoint="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
object_prefix="datasets/${dataset_slug}/${version}"
destination="r2:${R2_BUCKET}/${object_prefix}"

export RCLONE_CONFIG_R2_TYPE="s3"
export RCLONE_CONFIG_R2_PROVIDER="Cloudflare"
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="$endpoint"
export RCLONE_CONFIG_R2_NO_CHECK_BUCKET="true"

existing_objects="$(rclone lsf "$destination" --max-depth 1)"

if [[ -n "$existing_objects" ]]; then
    printf 'Refusing to overwrite published release: %s\n' "$destination" >&2
    printf 'Choose a new version or remove a known-incomplete upload in R2 first.\n' >&2
    exit 1
fi

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

while IFS= read -r -d '' file; do
    relative_path="${file#"$source_directory"/}"
    checksum="$(shasum -a 256 "$file" | awk '{print $1}')"
    printf '%s  %s\n' "$checksum" "$relative_path" >> "$manifest"
done < <(find "$source_directory" -type f -print0)

if [[ ! -s "$manifest" ]]; then
    printf 'Source directory contains no files: %s\n' "$source_directory" >&2
    exit 2
fi

rclone copy "$source_directory" "$destination" \
    --immutable \
    --progress

rclone copyto "$manifest" "${destination}/SHA256SUMS" \
    --immutable \
    --header-upload "Content-Type: text/plain" \
    --progress

printf 'Published %s %s to %s/\n' "$dataset_slug" "$version" "$destination"
if [[ -n "${R2_PUBLIC_BASE_URL:-}" ]]; then
    printf 'Public URL: %s/%s/\n' "${R2_PUBLIC_BASE_URL%/}" "$object_prefix"
fi