import argparse

from PINN_simulated_data import run_scenarios

### SEIR-PINN for synthetic data with sigma and gamma unknown: beta, sigma and gamma are all learned
### from observed I(t), using the PINN in PINN_simulated_data.py.
###
### sigma and gamma are learnable constants (started at 0.5; the true value is 0.25). 

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", default=["beta_0.75", "beta_0.5", "beta_0.4"])
ap.add_argument("--beta-const", action="store_true", help="beta is one learnable constant, not a network of time")
ap.add_argument("--split", type=float, default=1.0, help="fraction of days used for training")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--adam-steps", type=int, default=50_000)
ap.add_argument("--lbfgs", type=int, default=5000)
args = ap.parse_args()

run_scenarios(args.scenarios, args.split, args.seeds, args.adam_steps, args.lbfgs,
              learn_sg=True, beta_const=args.beta_const,
              prefix="PINN_all_learnable_parameters" + ("_beta_const" if args.beta_const else ""),
              pred_prefix="PINN_all_learnable" + ("_beta_const" if args.beta_const else ""))
