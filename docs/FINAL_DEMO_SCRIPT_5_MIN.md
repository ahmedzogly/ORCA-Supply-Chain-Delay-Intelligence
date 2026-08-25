# 5-Minute Live Demo Script

**Preparation:** 
Ensure both `python run.py --api` and `python run.py --dashboard` are running.
Navigate to http://localhost:8501.

---

## 1. Landing Page (30 seconds)
- **Visual Focus:** The "Delay Intelligence" title and the three colored Evidence Badges (REAL DATA, MODEL OUTPUT, SIMULATED SCENARIO).
- **Talking Point:** "Welcome. Today I'll demonstrate a research-validated decision intelligence prototype for global supply chains. It's important to note this is a prototype, not a production control tower."
- **Action:** Click the primary button: **Open Executive Control Tower**.

## 2. Executive Control Tower (60 seconds)
- **Visual Focus:** Portfolio KPIs and Risk Tier distribution chart.
- **Talking Point:** "Here we see a live monitoring view of 100 sample shipments. The system computes a calibrated probability of delay for every shipment. Notice that most shipments correctly fall into the 'Low Risk' bucket, which is typical for healthy operations."
- **Action:** Scroll down to the Priority Shipments table.
- **Talking Point:** "The system triages the portfolio, surfacing the highest-risk shipments. The highest-risk shipment in the current demo portfolio is **83922** at 46.7%."
- **Action:** Click **Investigate shipment 83922**.

## 3. Shipment Risk Explorer (90 seconds)
- **Visual Focus:** The three KPI blocks (Late Probability, Decision, Expected Delay/Uncertainty).
- **Talking Point:** "For shipment 83922, the CatBoost classifier estimates a 46.7% probability of delay. But probability isn't enough; we need severity. If late, the LightGBM models estimate a 15.8-day delay, with a 90% conformal prediction interval between 1.4 and 37.2 days."
- **Action:** Show the SHAP visualization chart.
- **Talking Point:** "Below, local SHAP values explain *why* the model scored this shipment high risk—for example, high vendor historical volume increases risk. Importantly, SHAP explains the model, it does not identify causal effects."

## 4. Decision & Action Center (60 seconds)
- **Visual Focus:** The SIMULATED SCENARIO badges and interactive sliders.
- **Talking Point:** "Knowing a shipment will be late is only valuable if we can act. In this Action Center, we map model outputs to business economics."
- **Action:** Move the 'Assumed intervention cost' slider from $500 to $200.
- **Talking Point:** "If an intervention costs $500, the scenario net benefit is -$150, leading to a 'MONITOR' recommendation. But if we secure a cheaper intervention at $200, the net benefit becomes positive ($150), and the policy recommendation instantly changes to 'INTERVENE'. These are simulated scenarios to stress-test planning assumptions, not accounting facts."

## 5. Portfolio Intelligence (30 seconds)
- **Action:** Navigate to Portfolio Intelligence via sidebar.
- **Visual Focus:** Severity distribution and Risk by Fulfillment Channel.
- **Talking Point:** "Zooming back out, decision-makers can view the macro risk distribution across channels and modes, allowing them to balance operational capacity."

## 6. Model Evidence (30 seconds)
- **Action:** Navigate to Model Evidence via sidebar.
- **Visual Focus:** The 5 academic tabs, starting with Predictive Performance.
- **Talking Point:** "Finally, academic and technical reviewers have full transparency. The PR-AUC is 0.27, representing a 4.5x improvement over a random baseline. We also transparently document empirical coverage, validation temporal design, and critical system limitations."
- **Conclusion:** "Thank you. The floor is open for questions."
