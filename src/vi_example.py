import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
import argparse
from utils import *


# Step 2: Mixture model
def model(data, K=1):
    with pyro.plate("components", K):
        alpha = pyro.sample("alpha", dist.Exponential(1.0))
        beta = pyro.sample("beta", dist.Exponential(1.0))

    mix_logits = pyro.sample("mix_logits", dist.Normal(torch.zeros(K), 1.0).to_event(1))
    mix_probs = torch.softmax(mix_logits, dim=0)

    with pyro.plate("data", len(data)):
        z = pyro.sample("z", dist.Categorical(mix_probs))
        pyro.sample("obs", dist.Gamma(alpha[z], beta[z]), obs=data)

# Step 3: Guide (no .to_event(1) here!)
def guide(data, K=1):
    alpha_q = pyro.param("alpha_q", torch.ones(K), constraint=dist.constraints.positive)
    beta_q = pyro.param("beta_q", torch.ones(K), constraint=dist.constraints.positive)
    mix_logits_q = pyro.param("mix_logits_q", torch.zeros(K))

    with pyro.plate("components", K):
        pyro.sample("alpha", dist.Delta(alpha_q))
        pyro.sample("beta", dist.Delta(beta_q))

    pyro.sample("mix_logits", dist.Delta(mix_logits_q).to_event(1))

if __name__ == "__main__":
  # Fix randomness
  pyro.clear_param_store()
  torch.manual_seed(42)

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filename', help="directory of your data")
  parser.add_argument('-l', '--local_ip', help="local ip address")
  args = parser.parse_args()
  
  df = load_csv(args.filename)
  # stats_src_ip(df)
  # stats_dst_ip(df)
  dl_pkt_interval = extract_dl_pkt_interval_time(df, args.local_ip)
  ul_pkt_interval = extract_ul_pkt_interval_time(df, args.local_ip)

  dl_pkt_interval = dl_pkt_interval[dl_pkt_interval < 0.2] * 1000 # in ms
  ul_pkt_interval = ul_pkt_interval[ul_pkt_interval < 0.2] * 1000 # in ms

  data = torch.from_numpy(dl_pkt_interval)

  # Step 4: Inference loop
  svi = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

  for step in range(3000):
      loss = svi.step(data)
      if step % 500 == 0:
          print(f"[Step {step}] ELBO: {loss:.2f}")

  # Step 5: Extract learned parameters
  alpha_q = pyro.param("alpha_q").detach()
  beta_q = pyro.param("beta_q").detach()
  mix_probs = torch.softmax(pyro.param("mix_logits_q").detach(), dim=0)

  print("\nEstimated Mixture Weights:", mix_probs)
  for i in range(len(alpha_q)):
      print(f"Component {i}: alpha = {alpha_q[i]:.2f}, beta = {beta_q[i]:.2f}")

  # Step 6: Plot histogram vs fitted mixture
  x = torch.linspace(0.01, data.max(), 10000)
  pdf = sum([
      mix_probs[i] * dist.Gamma(alpha_q[i], beta_q[i]).log_prob(x).exp()
      for i in range(len(alpha_q))
  ])

  dl_count, dl_bins_count = np.histogram(dl_pkt_interval, bins=10000) 
  dl_pdf = dl_count / sum(dl_count) 
  dl_cdf = np.cumsum(dl_pdf) 

  plt.plot(dl_bins_count[1:], dl_pdf, label="Data")
  plt.plot(x.numpy(), pdf.numpy(), label="Fitted Mixture", linewidth=2)
  plt.xlabel("Inter-arrival Time")
  plt.ylabel("Density")
  plt.title("Variational Inference on Gamma Mixture Model")
  plt.legend()
  plt.show()
