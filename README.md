In this Repository, I have added the codes for finding out the missing values in the NIFTY50's Implied Volatilities where the data is of almost per second frequency
Since, the dataset size exceeded, I couldn't add them in here but these codes can be used accordingly
I've tried various ways and tried to do something different since this was a contest in Kaggle, I decided to use the route of NOT using Gradient boosting here and rather try something new.
I tried experimenting with Physics Informed Neural Networks for options pricing.
According to Black Scholes Model, 
- Implied volatility is always non-negative
- Implied volatility is bounded in real markets (for NIFTY, values > 3 are essentially non existent)
- Volatility surfaces are smooth, not spiky or discontinuous
Allowing a neural network to violate these properties often leads to unstable training and poor generalization.
Hence, I embedded these constraints directly into the model.
Physical Constraints:
- Since volatility>0, we can enforce softplus acitivation function at the output layer
  Reasons are:
  - Gives +ve outputs
  - Avoids gradient issues associated with hard clipping or ReLU
  - Smooth and differentiable everywhere

Softplus activation Function- F(x) = ln(1+e^x)

- Upper Bound on Realistic Volatility Values:
  - As mentioned before extremely large volatility values are non existent for NIFTY50 options.
  So to prevent such predictions, a physics-based penalty term was added to the loss function:
      L(physics)​=E[max(−σ,0)2]+E[max(σ−3.0,0)2]
    This penalizes:
    -any negative volatility predictions
    -Volatility values exceeding a realistic upper bound
Unlike hard clipping, this is a soft constraint, allowing flexibility while discouraging violations.
Smoothness and Numerical Stability:
  -While no explicit PDE residual was enforced, smoothness and stability were encouraged through architectural and optimization choices:
      -SiLU (Swish) activations for smooth non-linear mapping
      -Batch Normalization to reduce internal covariate shift
      -Gradient clipping to prevent exploding gradients
      -Feature normalization to stabilize training dynamics
  Together, these act as implicit physical regularizers, encouraging the learned IV surface to be smooth and well-behaved.


Our Final Loss function was: L(total)​=L(MSE)​+λ⋅L(physics)
PS- A full Black–Scholes PDE residual was not enforced due to the absence of option price labels in the dataset. Instead, the model incorporated financially motivated constraints on implied volatility, which proved effective and stable within the hackathon setting.​
