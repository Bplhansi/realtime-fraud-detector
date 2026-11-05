# Two-minute demo script

1. **(15 s) The problem.** "Fraud is 0.25% of payments and must be caught before the payment
   completes. Python learns from the past; C++ scores each new payment in microseconds."
2. **(20 s) One command.** Show `docker compose up --build` finishing: the pipeline log with the
   test results table and the engine line `latency p99: ... ns`.
3. **(30 s) Score one payment.** In http://localhost:8000/docs, `POST /score` a 9,500 TRANSFER
   from a new account to a stranger. Point at the score, `block`, and the three reasons.
4. **(35 s) Review it.** Open the dashboard, pick the top case, read its reasons and the
   sender's recent activity (money came in, then cashed out at once). Click **Confirm fraud**.
5. **(20 s) Overview.** Alerts over time, p50 / p99 latency, reviewer-confirmed precision.
   Switch to the shifted file to show the drift warning.
6. **(10 s) Honesty.** "Synthetic data, time-based split, test set used once, results in the README."
