#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source ./set_env_persist.sh ENV_VAR_NAME TOKEN_VALUE
#
# Notes:
# - You MUST "source" this script to affect the current terminal session.
# - It will also persist the variable for future terminals by updating shell rc files.

if [[ $# -ne 2 ]]; then
  echo "Usage: source ./set_env_persist.sh ENV_VAR_NAME TOKEN_VALUE" >&2
  exit 1
fi

VAR_NAME="$1"
VAR_VALUE="$2"

# Basic validation for a shell variable name
if [[ ! "$VAR_NAME" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Error: '$VAR_NAME' is not a valid environment variable name." >&2
  exit 1
fi

# Escape value safely for shell export line
escape_for_double_quotes() {
  local s="$1"
  s="${s//\\/\\\\}"   # backslashes
  s="${s//\"/\\\"}"   # double quotes
  s="${s//\$/\\\$}"   # dollars
  s="${s//\`/\\\`}"   # backticks
  printf '%s' "$s"
}
ESCAPED_VALUE="$(escape_for_double_quotes "$VAR_VALUE")"

EXPORT_LINE="export ${VAR_NAME}=\"${ESCAPED_VALUE}\""
MARKER_BEGIN="# >>> managed-by-set-env-persist (do not edit) >>>"
MARKER_END="# <<< managed-by-set-env-persist (do not edit) <<<"

update_rc_file() {
  local rc_file="$1"

  # Create file if it doesn't exist
  if [[ ! -f "$rc_file" ]]; then
    touch "$rc_file"
  fi

  # If we already manage this file, replace the variable line inside the block
  if grep -qF "$MARKER_BEGIN" "$rc_file"; then
    # Replace any existing export line for this var inside the managed block; or insert if missing.
    awk -v var="$VAR_NAME" -v newline="$EXPORT_LINE" -v begin="$MARKER_BEGIN" -v end="$MARKER_END" '
      BEGIN { inblock=0; found=0 }
      {
        if ($0 == begin) { inblock=1; print; next }
        if ($0 == end) {
          if (inblock && !found) print newline
          inblock=0
          print
          next
        }
        if (inblock && $0 ~ "^export[ \t]+" var "=") { print newline; found=1; next }
        print
      }
    ' "$rc_file" > "${rc_file}.tmp" && mv "${rc_file}.tmp" "$rc_file"
  else
    # Append a managed block
    {
      echo ""
      echo "$MARKER_BEGIN"
      echo "$EXPORT_LINE"
      echo "$MARKER_END"
    } >> "$rc_file"
  fi
}

# Persist for future terminals (Bash + Zsh common files)
update_rc_file "$HOME/.bashrc"
update_rc_file "$HOME/.bash_profile"
update_rc_file "$HOME/.zshrc"

# Set for current terminal session (only works if sourced)
export "${VAR_NAME}=${VAR_VALUE}"

# If script is executed (not sourced), it can't affect the parent shell.
# Detect that and warn.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Note: This script was executed, not sourced, so it cannot set the variable in your current terminal."
  echo "Run: source ./set_env_persist.sh $VAR_NAME '<token>'"
else
  echo "Set for current session: $VAR_NAME"
  echo "Persisted for future sessions in: ~/.bashrc, ~/.bash_profile, ~/.zshrc"
fi