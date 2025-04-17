import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

# Fix randomness
pyro.clear_param_store()
torch.manual_seed(42)

# Step 1: Simulate inter-arrival time data (mixture of 2 Gamma distributions)
n_samples = 1000
true_weights = torch.tensor([0.6, 0.4])
true_alphas = torch.tensor([2.0, 6.0])
true_betas = torch.tensor([1.0, 2.0])
z = dist.Categorical(true_weights).sample([n_samples])
data = torch.stack([dist.Gamma(true_alphas[i], true_betas[i]).sample() for i in z])

# Step 2: Mixture model
def model(data, K=10):
    with pyro.plate("components", K):
        alpha = pyro.sample("alpha", dist.Exponential(1.0))
        beta = pyro.sample("beta", dist.Exponential(1.0))

    mix_logits = pyro.sample("mix_logits", dist.Normal(torch.zeros(K), 1.0).to_event(1))
    mix_probs = torch.softmax(mix_logits, dim=0)

    with pyro.plate("data", len(data)):
        z = pyro.sample("z", dist.Categorical(mix_probs))
        pyro.sample("obs", dist.Gamma(alpha[z], beta[z]), obs=data)

# Step 3: Guide (no .to_event(1) here!)
def guide(data, K=10):
    alpha_q = pyro.param("alpha_q", torch.ones(K), constraint=dist.constraints.positive)
    beta_q = pyro.param("beta_q", torch.ones(K), constraint=dist.constraints.positive)
    mix_logits_q = pyro.param("mix_logits_q", torch.zeros(K))

    with pyro.plate("components", K):
        pyro.sample("alpha", dist.Delta(alpha_q))
        pyro.sample("beta", dist.Delta(beta_q))

    pyro.sample("mix_logits", dist.Delta(mix_logits_q).to_event(1))

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
x = torch.linspace(0.01, data.max(), 300)
pdf = sum([
    mix_probs[i] * dist.Gamma(alpha_q[i], beta_q[i]).log_prob(x).exp()
    for i in range(len(alpha_q))
])

plt.hist(data.numpy(), bins=50, density=True, alpha=0.5, label="Data")
plt.plot(x.numpy(), pdf.numpy(), label="Fitted Mixture", linewidth=2)
plt.xlabel("Inter-arrival Time")
plt.ylabel("Density")
plt.title("Variational Inference on Gamma Mixture Model")
plt.legend()
plt.show()
