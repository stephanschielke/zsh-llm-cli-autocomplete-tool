#!/usr/bin/env zsh
# LLM Completion Strategy for zsh-autosuggestions

_zsh_autosuggest_strategy_llm() {
    emulate -L zsh
    setopt EXTENDED_GLOB
    
    # Must use global variable for zsh-autosuggestions
    typeset -g suggestion
    local buffer="$1"
    
    # Minimum length check
    local min_length="${ZSH_LLM_MIN_LENGTH:-3}"
    (( ${#buffer} < min_length )) && return
    
    # Debounce: Don't call if last call was <500ms ago
    local current_time=$(date +%s%N | cut -b1-13)  # milliseconds
    local last_time="${_ZSH_LLM_LAST_CALL_TIME:-0}"
    local debounce_ms="${ZSH_LLM_DEBOUNCE_MS:-500}"
    
    if (( current_time - last_time < debounce_ms )); then
        # Use cached value if available
        if [[ "$buffer" == "$_ZSH_LLM_LAST_BUFFER" ]]; then
            suggestion="$_ZSH_LLM_LAST_SUGGESTION"
        fi
        return
    fi
    
    # Cache check: Return cached if same buffer and <5 seconds old
    local cache_ttl="${ZSH_LLM_CACHE_TTL:-5}"
    if [[ "$buffer" == "$_ZSH_LLM_LAST_BUFFER" ]]; then
        local cache_age=$(( (current_time - _ZSH_LLM_LAST_TIME) / 1000 ))
        if (( cache_age < cache_ttl )); then
            suggestion="$_ZSH_LLM_LAST_SUGGESTION"
            return
        fi
    fi
    
    # Get daemon port (respect environment)
    local port="${MODEL_COMPLETION_DAEMON_PORT:-11435}"
    local timeout="${ZSH_LLM_TIMEOUT:-3}"
    
    # Call daemon
    local prediction
    prediction=$(printf '%s' "$buffer" | curl -s -X POST \
        --data-binary @- \
        --max-time "$timeout" \
        "http://127.0.0.1:${port}/complete" 2>/dev/null)
    
    # Validate prediction:
    # - Must not be empty
    # - Must not equal buffer
    # - Must start with buffer (match prefix)
    if [[ -n "$prediction" && 
          "$prediction" != "$buffer" && 
          "$prediction" == "$buffer"* ]]; then
        suggestion="$prediction"
    fi
    
    # Update cache on success
    if [[ -n "$suggestion" ]]; then
        _ZSH_LLM_LAST_BUFFER="$buffer"
        _ZSH_LLM_LAST_SUGGESTION="$suggestion"
        _ZSH_LLM_LAST_TIME="$current_time"
    fi
    _ZSH_LLM_LAST_CALL_TIME="$current_time"
    
    # Empty suggestion = fallback to next strategy
}