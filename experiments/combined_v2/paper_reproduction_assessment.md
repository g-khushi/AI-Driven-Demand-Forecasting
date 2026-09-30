# MARIOD reference assessment and comparison boundary

Source: user-supplied `sensors-25-02428-v2.pdf`, Yang et al., Sensors 2025, 25, 2428, DOI https://doi.org/10.3390/s25082428. Methodology and results pages 9–16 were extracted and visually inspected. The source is a reference, not an instruction overriding the user's experiment requirements.

## What the paper specifies

- Equations 9–11: historical-window, multi-horizon forecasting with pre-normalized Transformer attention and feed-forward residual blocks.
- Equation 12: separate query/key/value attention for each input modality.
- Equations 1–8 and 13–18: inventory state, order/reallocation actions, transport and inventory costs, hierarchical agents, and joint forecasting/inventory optimization.
- Algorithm 1: hierarchical PPO policy updates, critic, communication and replay buffers, joint objective.
- Sections 4.1.1–4.1.2: three retail sources, daily data, a stated 70/15/15 split, 512-dimensional attention with eight heads and six encoder/decoder layers; inventory actor/critic networks; training across eight 80-GB A100 GPUs.

The article discusses all three datasets but does not define a reproducible cross-dataset row union, mapping between retailer entities, common calendar, or combined metric aggregation. We must declare those decisions ourselves, preserve retailer identity and avoid adding incompatible sales units.

## Missing requirements for full MARIOD

The audited public data do not supply on-shelf/backroom/in-transit inventories, RFID/environmental/smart-shelf streams, competitor state, store–distribution-centre–corporate graph, reorder lead times/capacities, cost coefficients or action histories. The article says inventory records were added, without providing them in the supplied PDF. No official executable implementation was identified in the initial exact-title/code search or the article's availability statement. This is not proof that no implementation exists elsewhere.

The paper does not give a standalone forecast-residual correction algorithm or enough distributional output details to reproduce interval coverage/CRPS exactly. Error correction in our comparison must be an explicitly declared extension applied with the same opportunity to both methods. A forecasting-only reconstruction cannot be called full MARIOD and cannot support inventory or stockout claims.

## Important comparability limits

The paper's Table 2 reports one MARIOD row (MAPE 15.6%, RMSE 119.8, sMAPE 15.9%) without dataset-specific scoring rows, horizons, exact cohort IDs, zero-MAPE handling or prediction files. These published values are contextual reference only. They must not be ranked directly against our pilot errors, whose population and units differ. Its stated 18.2% reduction from MAPE 17.2 to 15.6 is inconsistent with those displayed numbers: the arithmetic gives about 9.3%.

The prior experiment's final-test boundaries will remain unchanged. A new 70/15/15 split would move already protected observations and would not recreate the unspecified published split. Missing sales remain conditionally zero only under our audited recording rules; unavailable sensors/inventories will not be fabricated.

## Work authorized and in progress

Combine the three existing training-selected 12-series cohorts in a labelled 36-series development panel, preserving raw datasets and prior results. Extend only through the remaining calibration folds. Improve the five-component Hybrid using bounded, development-only choices and test on later calibration origins. This is a bounded experiment, not full-population training.

The user has been asked to choose between a clearly labelled forecasting-only MARIOD reconstruction and supplying the authors' full implementation/additional data. Full MARIOD training and claims of beating full MARIOD are blocked by those missing inputs, not by a request for routine permission.
