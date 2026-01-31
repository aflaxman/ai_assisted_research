# PPV Calculator Notes

## GoatCounter Analytics

### Dashboard

View analytics at: https://aflaxman.goatcounter.com/

### What's Tracked

1. **Page views** - Automatically tracked when the page loads
2. **Calculations** - Custom event tracked when user interacts with the calculator (debounced to fire 500ms after interaction stops)

### Testing with DevTools

1. Open the calculator: https://aflaxman.github.io/ai_assisted_research/ppv_calculator/

2. Open DevTools (F12) and go to the **Network** tab

3. Filter by `goatcounter` to see only analytics requests

4. Interact with the calculator:
   - Drag a slider and release
   - Type in a number input
   - Click a prevalence cell in the table

5. After you stop interacting, you should see a request to `count` after ~500ms

### Testing Debounce in Console

Paste this in the Console to verify the debounced tracking function exists:

```javascript
typeof trackCalculation === 'function' ? 'Debounce is working!' : 'Function not found'
```

### Implementation Details

The telemetry uses a debounce pattern to avoid flooding GoatCounter with events when dragging sliders:

```javascript
let telemetryTimeout = null;
function trackCalculation() {
  clearTimeout(telemetryTimeout);
  telemetryTimeout = setTimeout(() => {
    if (window.goatcounter && window.goatcounter.count) {
      window.goatcounter.count({
        path: 'calculate',
        title: 'PPV Calculation',
        event: true
      });
    }
  }, 500);
}
```

This ensures each "interaction session" (dragging a slider, typing values) counts as one event rather than dozens.

### Ad Blockers

Some ad blockers block GoatCounter. For accurate testing, use a browser profile without ad blockers or add an exception for `gc.zgo.at`.
