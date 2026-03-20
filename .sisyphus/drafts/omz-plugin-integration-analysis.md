# Zsh Plugin Integration Analysis (REVISED)

## Critical Issues Identified by Metis Review

### Issue 1: Daemon Response Format
**DISCOVERY**: The current daemon returns a SINGLE prediction, not multiple.

Looking at daemon.py:
```python
result = complete(...)
self._send(result or command)  # Single string only
```

**IMPACT**: The plan for fzf menu with "top 5 predictions" is **NOT POSSIBLE** without daemon modification.

**SOLUTION OPTIONS**:
1. Modify daemon to support `?n=5` parameter returning JSON array
2. Accept single prediction only (simpler, less useful)
3. Call daemon multiple times with different parameters (inefficient)

### Issue 2: Variable Scope Bug
**CRITICAL**: Strategy functions MUST use `typeset -g suggestion` (global), not `local suggestion`.

```zsh
# WRONG:
local suggestion="$prediction"

# CORRECT:
typeset -g suggestion="$prediction"
```

### Issue 3: Race Conditions & Debouncing
**PROBLEM**: zsh-autosuggestions forks a subprocess per keystroke. If user types 14 chars rapidly = 14 parallel LLM calls.

**SOLUTION REQUIRED**:
- Implement debouncing (500ms delay after typing stops)
- Cancel pending requests on new keystroke
- Cache last buffer+prediction pair to avoid duplicate calls

### Issue 4: No State Management for fzf
**PROBLEMS**:
- When is prediction array cleared?
- What if buffer changes after prediction fetched?
- What if Tab pressed while strategy still fetching?

**SOLUTION**:
- Store timestamp with predictions
- Validate buffer matches before showing fzf
- Clear on accept-line, clear-line widgets

### Issue 5: POSTDISPLAY vs region_highlight
**CONFLICT**: 
- zsh-autosuggestions uses `POSTDISPLAY` for ghost text
- Current plugin uses `region_highlight`
- These fight each other = visual glitches

**DECISION**: Must use zsh-autosuggestions approach (POSTDISPLAY) for ghost line integration.

### Issue 6: Plugin File Location
**BAD**: Modifying files inside zsh-autosuggestions directory breaks on update.

**GOOD**: Create standalone OMZ-compatible plugin that registers with zsh-autosuggestions.

### Issue 7: Key Binding Conflicts
**COMPLEXITY**: Three plugins want Tab:
1. zsh-autocomplete (menu-select)
2. zsh-autosuggestions (clear on Tab)
3. Custom widget (fzf menu)

**SOLUTION**: Context-aware binding or separate trigger key.

## Questions for User (PRIORITY ORDER)

### BLOCKING (Must answer before implementation):

1. **Daemon modifications**: Should I modify the daemon to return multiple predictions (JSON array), or work with single prediction only?
   - Single: Simpler, just ghost line
   - Multiple: Enables fzf menu, requires daemon changes

2. **fzf menu approach**: If daemon returns single prediction, how should Tab work?
   - Option A: Tab always shows zsh-autocomplete menu (no fzf)
   - Option B: Add separate key binding (e.g., Ctrl+T) for fzf with history-based candidates
   - Option C: Modify daemon to support multiple predictions first

3. **LLM trigger timing**: When should LLM be called?
   - Option A: On every keystroke (like zsh-autosuggestions) - requires debouncing
   - Option B: After delay (e.g., 500ms after typing stops)
   - Option C: Only on specific trigger (e.g., Tab or key combo)

4. **Fallback priority**: If LLM is slow, what should user see?
   - Option A: Nothing until LLM returns (might feel sluggish)
   - Option B: Show history first, replace with LLM when ready (might jump)
   - Option C: Skip LLM if it takes >500ms, use history

### IMPORTANT (Should answer):

5. **Minimum buffer length**: How many characters before triggering LLM?
   - Current: 2 chars
   - Recommendation: 3-4 chars for better context

6. **Disable toggle**: Should there be a quick way to disable LLM suggestions?
   - Yes/No
   - If yes: key binding or command?

7. **Visual indicator**: Should we show when LLM is "thinking"?
   - Yes: Show spinner or "..." 
   - No: Stay silent

8. **Key binding for fzf**: What key should trigger the menu?
   - Tab (conflicts with zsh-autocomplete)
   - Ctrl+T (clean, no conflicts)
   - Ctrl+Space (currently used, but could be repurposed)
   - Other: _____

### NICE TO HAVE:

9. **Performance limits**: Max acceptable latency for LLM call?
   - Default: 3 seconds
   - User preference: _____

10. **Debounce time**: How long after typing stops before calling LLM?
    - Default: 500ms
    - User preference: _____

## Revised Architecture Options

### Option A: Single Prediction + Ghost Line (SIMPLE)
```
User types → zsh-autosuggestions strategy chain:
  1. llm_completion → returns 1 prediction → ghost line
  2. match_prev_cmd (fallback)
  3. history (fallback)
  4. completion (fallback)

Tab → Normal zsh-autocomplete behavior
Ctrl+T → fzf with history candidates (optional)
```

**Pros**: Simple, no daemon changes, clean integration  
**Cons**: No LLM-specific fzf menu

### Option B: Multiple Predictions + fzf (FULL FEATURED)
```
User types → zsh-autosuggestions strategy chain:
  1. llm_completion → returns 1 prediction → ghost line
     - ALSO stores top 5 in global array
  2. Other strategies as fallback

Tab → Custom widget:
  - If LLM predictions fresh → fzf with those
  - Else → normal zsh-autocomplete
```

**Pros**: Full feature set as requested  
**Cons**: Requires daemon modification, more complex

### Option C: Hybrid - History fzf with LLM priority
```
User types → zsh-autosuggestions strategy chain:
  1. llm_completion → ghost line only
  2. Other strategies populate fzf candidates

Tab → fzf with history/completion candidates
LLM just provides the "best" suggestion as ghost line
```

**Pros**: No daemon changes, uses existing fzf sources  
**Cons**: fzf doesn't show LLM alternatives

## My Recommendation

**Start with Option A** because:
1. Works with current daemon
2. Can be enhanced to Option B later
3. Immediate value without blocking on daemon work
4. Simpler = fewer bugs

Then **Option B as Phase 2** if user wants fzf menu with LLM predictions.

## Implementation Checklist

### Phase 1: Ghost Line Integration (Option A)

**Files to Create**:
1. `src/scripts/zsh_autosuggest_strategy_llm.zsh` - Strategy function
2. Update `src/scripts/zsh_autocomplete.plugin.zsh` - Integration hooks

**Files to Modify**:
1. `.zshrc` - Update ZSH_AUTOSUGGEST_STRATEGY order
2. `config/default.yaml` - Add debounce/timeout settings

**Key Implementation Details**:

```zsh
# Strategy function (src/scripts/zsh_autosuggest_strategy_llm.zsh)
_zsh_autosuggest_strategy_llm() {
    emulate -L zsh
    setopt EXTENDED_GLOB
    
    typeset -g suggestion
    local buffer="$1"
    
    # Skip if too short
    (( ${#buffer} < ${ZSH_LLM_MIN_LENGTH:-3} )) && return
    
    # Skip if recently fetched same buffer (caching)
    if [[ "$buffer" == "$_ZSH_LLM_LAST_BUFFER" ]] && 
       (( $(date +%s) - ${_ZSH_LLM_LAST_TIME:-0} < 5 )); then
        suggestion="$_ZSH_LLM_LAST_SUGGESTION"
        return
    fi
    
    # Call daemon
    local prediction
    prediction=$(printf '%s' "$buffer" | curl -s -X POST \
        --data-binary @- \
        --max-time ${ZSH_LLM_TIMEOUT:-3} \
        "http://127.0.0.1:${MODEL_COMPLETION_DAEMON_PORT:-11435}/complete" 2>/dev/null)
    
    # Validate
    if [[ -n "$prediction" && "$prediction" != "$buffer" && 
          "$prediction" == "$buffer"* ]]; then
        suggestion="$prediction"
        _ZSH_LLM_LAST_BUFFER="$buffer"
        _ZSH_LLM_LAST_SUGGESTION="$suggestion"
        _ZSH_LLM_LAST_TIME=$(date +%s)
    fi
}
```

### Phase 2: fzf Menu (Option B - Future)

**Requires**:
1. Daemon modification to support `/complete?n=5` returning JSON array
2. Custom widget for Tab binding
3. State management for prediction array

## Open Technical Questions

1. How does zsh-autosuggestions handle strategy timeout? Does it block or fall through?
2. Can we use zsh-autosuggestions' async infrastructure or need our own?
3. What's the cleanest way to add strategy without modifying zsh-autosuggestions files?
4. Should the strategy check if daemon is running before attempting call?

## Anti-Patterns to Avoid

- ❌ Modifying files in zsh-autosuggestions/ directory
- ❌ Using local instead of typeset -g in strategy
- ❌ Calling LLM synchronously in widget
- ❌ No debouncing on rapid keystrokes
- ❌ Not handling daemon-down scenario gracefully
