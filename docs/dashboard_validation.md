# Dashboard Validation

Testing ensures the dashboard accurately reflects underlying models without corrupting them.

## Safety Validations
- Features passed into the API explicitly drop Delay_Days and Delay_Flag via filtering loops, simulating blind real-world inference.
- Dashboard does not implement or duplicate decision rule thresholds.
- No session_state mutates raw parquet files.

## Scenario
A standard workflow spans Shipment Explorer -> Action Center, cleanly translating the predicted outputs into human approval gates.
