# Data

## Where it comes from

The project uses data in the format of **PaySim**, a public *synthetic* mobile-money dataset
([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)). PaySim is about 6.3 million rows,
so by default the project generates a smaller file in exactly the same format with
`python -m sim.generate` (no download or account needed). The real PaySim CSV works too: see
"Using the real PaySim file" below. Check PaySim's licence and description on Kaggle before use.

| Column | Meaning | Used? |
| --- | --- | --- |
| `step` | Time; 1 step = 1 hour (the month is steps 1-720 here; PaySim has 743) | yes |
| `type` | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` | yes |
| `amount` | Payment amount | yes |
| `nameOrig` | Sender, e.g. `C1231006815` (C = customer) | yes, as an id |
| `nameDest` | Receiver; `M...` = merchant | yes, as an id |
| `oldbalanceOrg`, `oldbalanceDest` | Balances before | **dropped**: inconsistent for fraud and merchants, hints at the label |
| `newbalanceOrig`, `newbalanceDest` | Balances after | **dropped**: not known at payment time, gives the label away |
| `isFraud` | The label | target only |
| `isFlaggedFraud` | PaySim's own rule (TRANSFER over 200,000) | **dropped**: another system's output |

## What the simulator does

- 25,000 customers and 2,500 merchants over 30 days. Each customer has a typical amount and
  activity level, pays merchants, cashes in and out, and sends money to 2-5 regular contacts.
- About 1% of customers make one legitimate large transfer to someone new (rent deposit,
  family help), so "a big transfer to a stranger" is not automatically fraud.
- 2% of customers have their account taken over: money is TRANSFERred to a **mule** account,
  which CASH_OUTs it within 3 hours. Each mule serves about three victims. A quarter of cases are
  small ("low and slow"), and 30% are split into two transfers. As in PaySim, both the transfer
  and the cash-out are labelled fraud.
- Balance columns behave like PaySim's, including the leak (fraud usually empties the account).

## What the data shows (`python -m ml.explore`)

On the default file (seed 7):

- **527,372 transactions, 1,296 fraud (0.246%)**: rare, so accuracy is meaningless (predicting
  "never fraud" is 99.75% accurate and catches nothing).
- Fraud appears **only in TRANSFER (1.01% of them) and CASH_OUT (0.82%)**; never in PAYMENT,
  DEBIT or CASH_IN. The same is true of the real PaySim data.
- Fraud is bigger: median amount **1,658** vs **122** for genuine payments.
- **The leak:** after the payment the sender's balance is exactly 0 for **91.0%** of fraud but
  only **12.3%** of genuine payments. A model allowed to see `newbalanceOrig` would look
  brilliant and be useless, because that number does not exist until after the decision.
- Fraud is spread evenly over the month (no seasonality), which keeps a time split fair.

The script also saves `reports/explore.png` (fraud per hour and amount distributions).

## Features (`config/features.txt`)

All computed from the account's history **before** the transaction (each transaction is scored,
then added to history). Each account remembers its last 64 outgoing and 64 incoming payments.

| Feature | Meaning |
| --- | --- |
| `amount_log` | log(1 + amount) |
| `amount_vs_sender_avg` | amount / the sender's average payment so far (1 if none) |
| `sender_tx_1h`, `sender_tx_24h` | sender's payments in the same hour / last 24 hours |
| `hours_since_sender_last` | hours since the sender's last payment (-1 if never) |
| `sender_is_new` | the sender has never paid anyone |
| `pair_seen_before` | the sender has paid this receiver before (within its last 64 payments) |
| `receiver_distinct_senders_24h` | how many different senders paid the receiver in 24 hours |
| `receiver_is_new` | the receiver has never been seen |
| `sender_inflow_24h_ratio` | amount / money the sender received in the last 24 hours (0 if none) |
| `hours_since_sender_first_seen` | age of the sender account in hours (-1 if new) |
| `receiver_is_merchant` | receiver id starts with M |
| `type_*` | one-hot transaction type |

## Split

Earliest 70% of rows for training, next 15% validation, final 15% test, cut on whole steps:

| Set | Steps (hours) | Rows | Fraud |
| --- | --- | --- | --- |
| Train | 1-507 | 369,185 | 916 |
| Validation | 508-614 | 79,683 | 219 |
| Test | 615-720 | 78,504 | 161 |

## Using the real PaySim file

1. Download `PS_20174392719_1491204439457_log.csv` from the Kaggle page and save it as
   `data/paysim.csv`.
2. Run the same steps with that file:

```bash
python -m ml.explore --csv data/paysim.csv
python -m ml.prepare --csv data/paysim.csv --out data
python -m ml.train && python -m ml.export_onnx
```

Feature building takes a few minutes for 6.3 million rows (it is a plain Python loop, kept simple
on purpose so it can be compared line by line with the C++ code).
