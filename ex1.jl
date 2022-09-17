### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ab655734-a8b7-47db-9b73-4099c4b11dfc
begin
	using CSV, DataFrames

	using LinearAlgebra, PDMats
	using Optim, ForwardDiff

	using Turing, Distributions
	using MCMCChains

	using Plots, StatsPlots, PairPlots, LaTeXStrings, ColorSchemes

	using PlutoUI, PlutoTeachingTools, HypertextLiteral

	# Set a seed for reproducibility.
	using Random
	Random.seed!(0)

	if false
		# Hide the progress prompt while sampling.
		#Turing.setprogress!(false);
		import Logging
		Logging.disable_logging(Logging.Warn)
	end
end;

# ╔═╡ 1c651c38-d151-11ec-22a6-87180bd6546a
md"""
**Astro 497, Lab 5, Ex 1**
# Model Building III:  Probabilistic Programming Languages
"""

# ╔═╡ d78db46d-1e20-4cd1-be53-81f635561b27
TableOfContents()

# ╔═╡ a6966023-77c4-4744-9950-3cc27d0061e1
md"""
## Overview

In previous labs, we fit models to measurements of transit times based on finding the model parameters that maximized the likelihood of a statistical model. 
We explored multiple physical models (i.e., simple linear model and sum of a linear model and a sinusoidal perturbation). 
But we stuck with one simple statistical model (i.e., independent Gaussian measurement uncertainties with known variance).
If we wanted to change our model for the measurement uncertainties, we would have to derive and implement an alternative likelihood function.  
While that's certainly possible, it can be time-consuming, tedious and error prone.  
It would be much nicer if we could simply state our model assumptions and have the computer figure out how to compute the likelihood for us.  
In this lab, you'll learn how to do that.

First, we'll demonstrate how to use a probabilistic programming language (PPL) to implement a simple model for a familiar datseta.  
Then, you'll see how we can use the PPL to easily revise the model to consider alternative choices for the prior distribution and/or likelihood.   
You'll get a chance to compare the posterior samples and posterior predictive distributions based. 

The primary purpose of this lab is for you to appreciate how using a PPL provides greater modeling flexibility than deriving results analytically.  
Thus, you can efficiently explore your data using multiple statistical models to better understand the sensitivity of your results to the inevitable assumptions.  
"""

# ╔═╡ 20eb786a-f042-429f-a5de-f1af8f69b2fc
md"""

## Ingest the data
To keep things simple, we'll start by reanalyzing measurements of the transit times of the planet Kepler-26b (also known as KOI 250.01).
"""
#  from [Holczer et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJS..225....9H/abstract).   

# ╔═╡ d49b2d7a-fabe-4525-a11c-8e808960fc96
begin
	data_path = "data/koi250.01.csv" # Data from Mazeh et al. 2013 ApJS 208, 16 (Table 2)
	df = CSV.read(data_path, select=[:n, :t, :σₜ], DataFrame) # t is BJD-2454900
	df.σₜ ./= 24*60   # convert units of uncertainties from minutes to days
	nobs = size(df,1)
	df
end

# ╔═╡ 90ae54f1-c49d-4d66-82eb-eb783c389e54
md"""
### Plot the data
Normally, we'd do a quick visualization, but since you've done that in previous labs, we'll only show that if you want to by checking this box
$(@bind show_plot_raw_data CheckBox(default=false)).
"""

# ╔═╡ efcabcc4-40ed-4492-b3e4-e190be248fce
if show_plot_raw_data
	scatter(df.n, df.t, yerr=df.σₜ, label=:none, xlabel="Transit #",  ylabel="Time (d)" )
end

# ╔═╡ 0873bbd9-f58b-4ec0-9506-15331f4a52fd
if show_plot_raw_data 
	md"""
Good call. It's always good to make a quick plot to make sure we understand the basic characteristics of the data.  For this data set, the transit times ($t$) are very nearly a linear function of the transit number ($n$).  While the measurement uncertainties are plotted, it's hard to see them in this plot.  
"""
end

# ╔═╡ c0e7a487-e8eb-4078-a473-e5fe64245f4c
md"""
### Ordinary linear regression
In order to have something to compare our results to, we'll go ahead and compute the best-fit linear model ($t_n = t_0 + n × P$), assuming independent Gaussian measurement noise with known variances ($\sigma_n$'s).  
Recalling [lab 3](https://psuastro497.github.io/lab3-start/ex1.html), this is a linear model:
```math
 \left[ \begin{matrix} t_1   \\ t_2  \\  \vdots  \\  t_{n_{nobs}} \end{matrix} \right]
=
\left[ \begin{matrix}
	1 & n_1  \\
	1 & n_2 \\
    \vdots  & \vdots  \\
    1 & n_{nobs}
 \end{matrix} \right]
\left[ \begin{matrix} t_0 \\ P \end{matrix} \right]
```
where the left-hand side contains the observed transit times, and the right-hand side includes the design matrix and the two parameters to be estimated.  
More generally, we could write this as
$y_{\mathrm{obs}} = \mathbf{A} b$.  

Recall that we can compute the maximum likelihood estimates of $b$ accounting for a given covariance matrix ($\Sigma$), using
$b_{mle}  = (A' {\Sigma}^{-1} A)^{-1} (A' \mathbf{\Sigma}^{-1} y_{obs})$.
In this case, the covariance matrix, $\mathbf{\Sigma}$, contains $\sigma_{t}$'s along the diagonal (allowing for more efficient calculations than if it were an arbitrary positive definite matrix).
"""

# ╔═╡ a77dd7a7-c9f7-4a93-9b85-207f88196650
begin
	A = [ones(nobs) df.n]
	covar = PDiagMat(df.σₜ)
	coef_mle_linalg_w_covar = Xt_invA_X(covar, A) \ (A' * (covar \ df.t) )
end

# ╔═╡ 76904349-d28e-4f4f-8867-e5c4c73c3b19
md"""
Since the above calculations reduce to linear algebra, they are very fast (for small to moderate-sized data sets like this one).  However, there are some important limitations.

**Q1a:**  Explain in your own words at least two reasons why one might need a more flexible approach, even for a data set where the underlying physics is linear?
"""

# ╔═╡ d88af6ab-f9c1-4aec-b335-106a3ab6ea45
response_1a = missing

# ╔═╡ be82ab95-65b0-4de7-b406-14b796065087
md"""
!!! hint "Hint"
    - What if we wanted to compute the maximum *a posteriori* value of $b$, $b_{map}$, while including prior information about the values of $t_0$ and $P$?  
	- What if we wanted to allow for measurement errors that are non-Gaussian?  
"""

# ╔═╡ 504f77c4-41db-4f70-b402-380fde67410d
md"""
# Probabilistic Programming Languages
**Probabilistic Programming Languages (PPLs)** make it easy for one to specify complex models.  There are many PPLs.  We'll be using [Turing.jl](https://turing.ml/dev/docs/using-turing/quick-start) below.
"""

# ╔═╡ 67ec0992-dd54-4172-b198-84bd61b7f48b
aside(md"""
!!! tip "Other PPLs"
    - [STAN](https://mc-stan.org/) has interfaces for Julia, Python, R and the command line),
    - [PyMC3](https://docs.pymc.io/en/v3/) and [numpyro](https://num.pyro.ai/en/stable/) are two of several options for python users,
    - [Nimble](https://r-nimble.org/) for R users, and
    - [Soss.jl](https://github.com/cscherrer/Soss.jl) is another option for Julia users, and
    - many others (e.g., BUGS and JAGS were early PPLs but rarely used for new projects today).

!!! tip
    STAN has a *great* [users guide](https://mc-stan.org/users/documentation/) that provides lots of good advice for Bayesian modeling regardless of how you do you computations.
""")

# ╔═╡ e5aa5c21-4f7b-4011-bb7d-a3e73d3de0a5
md"""
In PPLs, one doesn't specify how to compute the target distribution (e.g., the posterior probability distribution).  Instead, one specifies the distribution of all stochastic variables in a model using a high-level modeling language.  A compiler figures out how the computer can perform the necessary calculations.  

While there are inevitably differences in capabilities, efficiency and syntax, most PPLs adopt the common notation:
`x ~ Distribution`
to indicate that the random variable $x$ is drawn from a given distribution.  PPLs include many common distributions like `Uniform(a,b)`, `Normal(μ,σ)`, etc.  (Many modern PPLs now allow you to implement your own distributions, too.)

Some PPLs go so far as using a *[declarative programming](https://en.wikipedia.org/wiki/Declarative_programming)* model (as opposed to an imperative programming model where one specifies each step of the calculation in order).  This can create some limitations (e.g., such code can't interoperate with functions to perform general mathematical calculations).  Turing takes an approach where the order of commands does matter (e.g., you can only access a variable after it has been assigned a value or a distribution).  Still, Turing figures out the likelihood and how to sample from the prior and posterior probability distributions for you.
"""

# ╔═╡ 9e5c41c2-70c9-4b2b-84d0-dc223c818b31
md"""
For our first model, we'll implement the following model:

$t_0 \sim \mathrm{Uniform}(0,1200)$
$\mathrm{period} \sim \mathrm{Uniform}(1,100)$
$\mathrm{t}_i \sim \mathrm{Normal}(t_0 + \mathrm{period} \cdot n_i, \sigma_{t,i}^2)$

Inspect the code below for a Turing model for Bayesian linear regression.
"""

# ╔═╡ b8ab2460-7c59-4e87-8580-e7b49f0576aa
@model function linear_regression(x, y, σ_y)
    # Specify priors for model parameters
	t0 ~ Uniform(0, 1200)               # Time of 0th transit
    period ~ Uniform(1,100)             # Orbital period
	# Specify likelihood
	for i ∈ eachindex(y)
		t_pred = t0 + period * x[i]     # Predicted transit time given t0 and period
    	y[i] ~ Normal(t_pred, σ_y[i])   # Measurement model
    end
end

# ╔═╡ 3221c380-e04f-4822-ac5f-48add15aa757
md"""
Our model can be divided into three parts.
1.  We specify [prior probability](https://storopoli.github.io/Bayesian-Julia/pages/02_bayes_stats/#bayesian_statistics) distribution for each of the model parameters (in this case `t0` and `period`) on lines 3 & 4.
1. We specify the prediction of our model for each observation on line 7.
1. We specify the model for the probability of each measurement (`y[i]`, in this case observed transit times) conditioned on knowing the true values (in this case true transit times) on line 8.
"""

# ╔═╡ 77799e4f-0aea-4fd4-9a85-a22eba5ab2d4
md"""
Please pause to note a few things about the implementation of our model:
1.  We had to specify prior distributions for our model parameters, `t0` and `period`.  Sometimes just writing your model in a PPL can be helpful because it forces you to be explicit about all your assumptions.  
2.  Our model is a function that takes argument `x`, `y` and `σ_y`.  For our application, these will be arrays containing the transit number, observed transit time and measurement uncertainties.
3.  Our model includes standard Julia code, such as the `for` loop over observations and calculating `t_pred`.  (While most PPLs allow for deterministic nodes like `t_pred`, many PPLs are much more restrictive than Turing.)
"""

# ╔═╡ 81c18249-5951-48d9-a194-7f8e814cf8a3
md"""
In order to get a model that is conditioned on our observational data, we call the model function and pass the transit numbers, observed transit times and measurement uncertainties contained in the dataframe, `df`.
"""

# ╔═╡ 9926a73d-c8d0-4d83-9720-4313d82532bc
model_given_data = linear_regression(df.n, df.t, df.σₜ);

# ╔═╡ 3f1ee690-d9d0-4d39-ae3a-c7c42e7f0798
protip(md"""If we wanted to make it even more convenient, we could define a helper function that takes only a DataFrame.
```julia
linear_transit_time_model(df::DataFrame) = linear_regression(df.n, df.t, df.σₜ)
```
and call it like
```julia
linear_transit_time_model(df)
```
"""
)

# ╔═╡ 86cb312e-84df-42a4-bad5-2d74abca2560
md"""
## Best-Fit Model
Now we can use the model in a variety of ways.  For example, we can compute the maximum likelihood estimate and/or maximum *a posterior* parameters.  While not required, we'll provide a not-too-bad initial guess, so the calculations will be a little faster and reduce risk of bad convergence.
"""

# ╔═╡ c142f016-4b8d-4035-9dde-6c802c334546
init_guess = [ 100.0, 10.0]

# ╔═╡ 93ba5f00-a494-4c1c-8690-d50e3b398a7c
mle_estimate = optimize(model_given_data, MLE(), init_guess, GradientDescent())

# ╔═╡ 4ffc0147-9477-4cfd-acc5-5ca7994caf20
md"""
We can access the fit model parameter values using syntax like
`mle_estimate.values[1]` or `mle_estimate.values[:period]`.
Since our model might contain many parameters, it would be easy to lose track of which index is which variable.  Therefore, it's generally advised to refer to variables by name rather than by index.
"""

# ╔═╡ bf5e9688-993e-44fe-b6f0-6ab5daf2a4d8
begin	
	period_mle = mle_estimate.values[:period]
	t0_mle = mle_estimate.values[:t0]
end;

# ╔═╡ 9355219a-8911-4ab9-ae68-5891743280b3
md"""
At the bottom of the notebook, there's a helper function, `mode_result_to_named_tuple` to return a NamedTuple` based on the results of the optimize function.  That can be used as shown in the next cell.
"""

# ╔═╡ 3be23dcd-fd6d-40d0-a1ec-2eccc6222de3
md"""
Next, you'll repeat the above calculation, but computing the maximum *a posteriori* parameter estimate, rather than the MLE.  First, think about what you expect for the result.  

**Q2a:** How similar or different will the MAP and MLE estimates be for this particular model?
"""

# ╔═╡ da63ae7c-1a40-4091-8aff-ef7de62c3bb7
response_2a = missing 

# ╔═╡ 8317710d-50b5-4958-9a85-611133fd6ab8
if !ismissing(response_2a) 
	map_estimate = optimize(model_given_data, MAP(), init_guess, GradientDescent()) 
end

# ╔═╡ 99bd8a04-9b1c-4bcc-a87b-01e0feeed341
md"""
**Q2b:** Compare the MLE and MAP values.  If the results differed from your prediction, explain why.
"""

# ╔═╡ 507395bd-a56c-4cdb-b7cd-182dcc9b1a34
response_2b = missing

# ╔═╡ 62f0b1e4-2885-433c-a13d-c87d6ae2fd9b
md"Now, we can evaluate the log prior and log likelihood for any set of model parameters using `logprior` and `loglikelihood`."

# ╔═╡ 763a4a55-3f2f-46d6-98e7-f66327a2dc29
md"""
The joint probability refere to the product of the prior and likelihood and can be accessed via `logjoint`.
"""

# ╔═╡ b3363ef1-f046-49f3-b096-ad8cfb81e7f2
md"""
And we can plot contours of the log prior, log likelihood, or log joint proability.  
"""

# ╔═╡ d9d67f4f-01db-453a-bf69-5454a432b5fd
md"""
## Testing sensitivity to prior
Based on what we've seen so far, it may seem that a PPL is primary syntactic sugar.  Everything we've done so far could have been done reasonable some other way.  So why are they useful?  
PPLs can shine when you want to:
- Analyze data using a prior that makes the model non-linear,
- Analyze the data using a different model for the measurement errors,
- Test the sensitivity of your results to the choice of prior and/or noise model, or
- Use a model for which it would be really annoying to try to derive the log likelihood analytically, or
- Explore several variations on your model rapidly (talking about human time, not computer time).

For example, let's say we wanted to try a different prior for the orbital period.  We know there are some planets with orbital periods less than one day, so choosing a prior that assigns zero probability to such planets is dangerous.  

Based on previous studies, we also know that the occurrence rate of planets around sun-like stars doesn't drop to zero at 100 days.  Instead, it has a local maximum and decreases gradually.  So instead, we could pick a [log normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution) for our period prior.
"""

# ╔═╡ b4fb0106-7981-4349-85b0-09e2bd6e48d8
alt_period_prior = LogNormal(log(365), 1);

# ╔═╡ 265123e6-f9f2-44f6-b1db-35f903ceaa9f
md"""
Below, we'll visually inspect the two priors for orbital period.
"""

# ╔═╡ 63e72a21-24c3-410e-b8da-6c5abcfebb43
let
	minx_plt, maxx_plt = 0, 365
	plt =  plot( Uniform(1,100), label="Uniform Prior")
	plot!(plt, alt_period_prior, xlims=(minx_plt, maxx_plt), ylims=(0,maximum(pdf.(alt_period_prior,range(minx_plt,stop=maxx_plt, length=100)))), xlabel="Period (d)", ylabel="Prior PDF", label="Log Normal Prior")
	ylims!(0,0.011)
end

# ╔═╡ fe5acfd9-fb85-4432-8f7e-90162fdeb5f7
md"""
Deriving the MAP with this new prior would take a fair bit of algebra.  Instead, we can easily build a second model, swapping out the period prior by changing one line of code (line 4).
"""

# ╔═╡ 898cda4c-e193-4fdc-8139-7bb6ec858da0
@model function linear_regression_alt_prior(x, y, σ_y)
    # Specify priors for model parameters
	t0 ~ Uniform(0, 1200)               # Time of 0th transit
    period ~ alt_period_prior           # Alternative prior for orbital period
	# Specify likelihood
	for i ∈ eachindex(y)
		t_pred = t0 + period * x[i]     # Predicted transit time given t0 and period
    	y[i] ~ Normal(t_pred, σ_y[i])   # Error model
    end
end

# ╔═╡ 89fdecd6-57fd-4091-ac06-28edf04d0d01
model_given_data_alt_prior = linear_regression_alt_prior(df.n, df.t, df.σₜ)

# ╔═╡ 010b2e32-6bd4-47fa-a9c9-a0e14db92d8f
md"""
**Q2c:**  How similar or different do you expect the MAP estimates using the two choices of period prior will be?  
"""

# ╔═╡ 5c4b0ec6-f684-4293-899a-339885af4e36
response_2c = missing

# ╔═╡ 1d526aee-7d67-4688-a163-7c3cffa3d6f3
if !ismissing(response_2c)
	map_estimate_alt_prior = optimize(model_given_data_alt_prior, MAP(), init_guess, ConjugateGradient())
end

# ╔═╡ f93d7f0e-02b0-4935-9f5a-ab93ca28604b
aside(md"""
!!! tip "Tweaking Optimization Algorithm"
    Sometimes you may get warning messages about convergence.  In this case, the results are already good enough for our purposes.  If you wanted to dig deeper, the [Turing.jl user guide](https://turing.ml/dev/docs/using-turing/guide#maximum-likelihood-and-maximum-a-posterior-estimates) shows how you can choose your optimization algorithm and pass optional configuration parameters to improve the rate and chances of convergence.
""", v_offset=-120)

# ╔═╡ 2e433784-fd00-4fe9-a835-b37b7ec94b7c
md"""
**Q2d:**  How did the results compare to your predictions?  How similar or different do you expect the MAP estimates using the two choices of period prior will be?  
"""

# ╔═╡ 1993a166-9bea-4ded-9f12-5deb1327c133
md"""
Add contours for log joint probability using alternative prior to contour plot above by checking this box $(@bind show_joint_prob_alt_prior CheckBox(default=false)).
"""

# ╔═╡ e8c0f93a-ebfb-47cb-bf88-5c2c66483683
md"""
**Q2e:**  Inspect the contours for the log joint probability using the two priors.  What is the most significant difference in our understanding of the planet's orbit under the two choices of prior?
"""

# ╔═╡ 4ca9f2d6-d88d-49d5-8f1a-f8d464c7f7b9
response_2e = missing

# ╔═╡ e746194b-63dc-444e-92cc-a2dbecdd6d2f
md"""
### Sensitivity to error model
Similarly, we can easily swap out the likelihood used for the observations to make different assumptions for the distributions of observations measurements relative to the predicted transit time.  In this case, we'll explore a model where we assume that each observation is affected by a combination of the reported measurement noise and an additional "jitter" which we'll model as i.i.d. Gaussian with unknown variance.  
"""

# ╔═╡ 2e15c59a-d990-453f-baa5-19128573df02
@model function linear_regression_jitter(x, y, σ_y)
    # Specify priors for model parameters
	t0 ~ Uniform(0, 1200)               # Time of 0th transit
  period ~ alt_period_prior           # Alternative prior for orbital period
	# Specify liklihood
	hour = 1/24
	σ_jitter ~ LogNormal(hour, log(2))  # Prior for jitter
	for i ∈ eachindex(y)
		t_pred = t0 + period * x[i]     # Predicted transit time given t0 and period
    	y[i] ~  Normal(t_pred, sqrt(σ_jitter^2+σ_y[i]^2))
    end
end

# ╔═╡ eea61c7a-8dde-4a00-8b67-1b0adda052cf
linear_regression_jitter

# ╔═╡ 997dacbf-e0b1-4aac-977f-dd86180eb7dd
model_given_data_jitter = linear_regression_jitter(df.n, df.t, df.σₜ);

# ╔═╡ 09b185e3-a166-4168-81e4-2e9fb8a3132b
map_estimate_jitter = optimize(model_given_data_jitter, MAP(), vcat(mle_estimate.values.array,(10/(24*60))) )

# ╔═╡ 041aad0e-52e8-4ca2-80b1-24863d129ae4
md"""
**Q3a:** Compare the MAP estimates for the orbital period and t0.  Are conclusions about the orbital period more sensitive to the choice of prior or the choice of likelihood?
"""

# ╔═╡ 83f51ee0-eda6-4e14-83b1-7b76497e5b44
response_3a = missing

# ╔═╡ f5048ccb-c549-483c-af8b-20482bac5cb6
!ismissing(response_3a) || still_missing()

# ╔═╡ d1564b23-fdb7-499a-bab4-05363e6e7f53
md"""
### Improving the model
Based on the previous analysis, it is clear that our original model was pretty good, but mathematically it is not a good description of the data at the level of ~10 minutes.  Let's plot the residuals of the observed transit times relative to the posterior predictive distribution.
"""

# ╔═╡ 885ecd47-1093-4fe3-8e23-f5da27cce279
predict_linear(n, θ) = θ.t0 .+ θ.period .* n

# ╔═╡ cd862907-3590-4b05-8715-797c2edae0bc
md"""
Ah ha!  There's an additional signal.  In this case, the transit times aren't strictly linear because of gravitational perturbations from to another planet in this planetary system.  
A detailed model of n-body dynamics is computationally quite expensive.  However, we can easily try a modeling the transit times as the sum of a linear emphemeris and an additional sinusoidal signal.  
"""

# ╔═╡ 969c184a-1a7f-4c3d-a33e-6bc8e61f8e96
function predict_linear_plus_sinusoid(n,θ) 
		mu0 = θ.t0 .+ n .* θ.period
		mu  = mu0 .+ θ.ttv_amplitude_sin .* sin.((2π/θ.ttv_period).*mu0) .+ θ.ttv_amplitude_cos .* cos.((2π/θ.ttv_period).*mu0)
end

# ╔═╡ f379a950-18f2-439f-8c36-53ce8e461247
# Bayesian linear regression.
@model function linear_plus_sinusoid(x, y, σ_y)
    t0 ~ Uniform(0.0, 1200.0)
    period ~ alt_period_prior
	ttv_period ~ alt_period_prior
	ttv_amplitude_sin ~ Normal(0.0, 0.1)
	ttv_amplitude_cos ~ Normal(0.0, 0.1)
	mu = predict_linear_plus_sinusoid(x, (;t0, period, ttv_period, ttv_amplitude_sin, ttv_amplitude_cos) )
	y ~ MvNormal(mu, σ_y)
end

# ╔═╡ 70a61c2b-4292-417b-b306-ff09ad112f52
model_given_data_ttv_model = linear_plus_sinusoid(df.n, df.t, df.σₜ);

# ╔═╡ efe787bc-5af9-4a45-9e4f-bb503ea83b22
begin
	P_ttv_guses = 800.0
	A_guess = -5/(60*24)
	B_guess =  5/(60*24)
	init_guess_for_ttv_model = [ t0_mle, period_mle, P_ttv_guses, A_guess, B_guess ]
	
	map_estimate_ttv_model = optimize(model_given_data_ttv_model, MAP(), init_guess_for_ttv_model,  Optim.Options(f_tol=1e-5) )
end

# ╔═╡ 78a2895d-b059-469a-b3a7-1bacf7248757
md"""
This demonstrates the power of PPLs.  It's easy to rapidly try out different assumptions and models.  Finally, we'll combined each of our model improvements.
"""

# ╔═╡ 603a8abe-2e31-43f4-91ed-46857b65d089
# Bayesian linear regression.
@model function linear_plus_sinusoid_and_jitter(x, y, σ_y)
    t0 ~ Uniform(0.0, 1200.0)
    period ~ alt_period_prior
	ttv_period ~ alt_period_prior
	ttv_amplitude_sin ~ Normal(0.0, 0.1)
	ttv_amplitude_cos ~ Normal(0.0, 0.1)
	σ_j ~ LogNormal(log(1/(24*60)),1)
	pred = predict_linear_plus_sinusoid(x, (;t0, period, ttv_period, ttv_amplitude_sin, ttv_amplitude_cos) )
	for i ∈ eachindex(y)
		y[i] ~ Normal(pred[i],sqrt(σ_y[i]^2+σ_j^2))   # Measurement model
	end
	y
end

# ╔═╡ 3fdf576d-abb6-4ef5-a1fb-1877fb122cc1
model_given_data_ttv_model_jitter = linear_plus_sinusoid_and_jitter(df.n, df.t, df.σₜ);

# ╔═╡ f364ffc7-ce15-4bf1-ae80-f797c9d5f42b
begin
	σ_j_guess = 1/(60*24)
	init_guess_for_ttv_model_with_jitter = [ t0_mle, period_mle, P_ttv_guses, A_guess, B_guess, σ_j_guess ]
	
	map_estimate_ttv_model_jitter = optimize(model_given_data_ttv_model_jitter, MAP(), init_guess_for_ttv_model_with_jitter, Optim.Options(f_tol=1e-5) )
end


# ╔═╡ dcad1cad-f2c8-414e-9fed-e0de340e0603
md"""
**Q3b:** Compare the MAP estimates for the orbital period and t0 using the model that includes a sinusoidal TTV signal (and a jitter term) to the comparable model without a TTV signal.  Are the conclusions about the orbital period and t0 more sensitive to the choice of prior & likelihood or the choice of model?
"""

# ╔═╡ acb2f396-75e4-4475-93d8-54c72fd64189
response_3b = missing

# ╔═╡ 456f136d-1017-4a00-bc41-b8fcd4a6004b
!ismissing(response_3b) || still_missing()

# ╔═╡ 88be73c3-8049-489a-a265-254e29763be6


# ╔═╡ a480bd7a-a4a9-43c7-81f5-77d15ec16289
md"""
# Setup & Helper Code
"""

# ╔═╡ 13a03add-4262-4bd0-9dc7-426a57b8e0e4
mode_result_to_named_tuple(x::Turing.ModeResult) = (; zip(first(names(x.values)), 
   values(x.values))... )

# ╔═╡ 24f0e74c-603d-4eb5-a8a7-1f602941a858
param_mle = mode_result_to_named_tuple(mle_estimate)

# ╔═╡ 8114804f-8ca8-43f0-b1d7-7c014270a77e
logprior(model_given_data, param_mle )

# ╔═╡ 308c16b2-0eb4-4df5-9436-ed6d6908f58c
max_log_likelihood = loglikelihood(model_given_data, param_mle)

# ╔═╡ a9a7eca3-0562-42f4-9215-a877b6b2069a
if (@isdefined map_estimate)
	param_map = mode_result_to_named_tuple(map_estimate)
	(; Δperiod = param_map.period - param_mle.period, 
	  Δt0 = param_map.t0 - param_mle.t0)
end;

# ╔═╡ bda05df7-0bc1-4025-88ac-109c0136618f
if @isdefined param_map
max_log_joint = logjoint(model_given_data, param_map )
end

# ╔═╡ 325a5daf-e849-443c-95d8-630fe2d850c2
if @isdefined param_map
let
    x = range(param_map.t0-0.002, stop=param_map.t0+0.002, length=80)
    y = range(param_map.period-0.00005, stop=param_map.period+0.00005, length=80)
	scatter([param_mle.t0],[param_mle.period], mc=:red, markershape=:x,label=:none)
    contour!(x, y, (x,y)-> logjoint(model_given_data, (;period=y, t0=x ) ), levels=range(max_log_joint, step=-1, stop=max_log_joint-10), c=:red, colorbar=false)
	if show_joint_prob_alt_prior
		contour!(x, y, (x,y)-> logjoint(model_given_data_alt_prior, (;period=y, t0=x ) ), levels=range(max_log_joint, step=-1, stop=max_log_joint-10), c=:blue  )
		scatter!([param_map.t0],[param_map.period], mc=:blue, markershape=:+,label=:none)
    
	end
	xlabel!("t₀ (d)")
	ylabel!("Period (d)")
	title!("log Joint Probability = log Prior + log Likelihood")
end
end

# ╔═╡ 92ce947a-9f57-4efa-9b26-9d5e9efbd814
if !ismissing(response_2c)
	(;Δperiod = mode_result_to_named_tuple(map_estimate_alt_prior).period-param_map.period, 
	Δt0 = mode_result_to_named_tuple(map_estimate_alt_prior).t0-param_map.t0 )
end

# ╔═╡ ad96b064-e7a4-4b43-8368-56ae3e6059a2
if @isdefined(map_estimate_alt_prior)
	(;Δperiod = 
		mode_result_to_named_tuple(map_estimate_jitter).period-
		mode_result_to_named_tuple(map_estimate_alt_prior).period, 
	Δt0 = mode_result_to_named_tuple(map_estimate_jitter).t0 -
		mode_result_to_named_tuple(map_estimate_alt_prior).t0 )
end

# ╔═╡ a91ef793-3c94-4f14-be04-d78d5f7106b5
if (@isdefined map_estimate_alt_prior) && (@isdefined map_estimate_ttv_model )
(; Δperiod = mode_result_to_named_tuple(map_estimate_ttv_model).period - mode_result_to_named_tuple(map_estimate_alt_prior).period, 
   Δt0 = mode_result_to_named_tuple(map_estimate_ttv_model).t0 -
		mode_result_to_named_tuple(map_estimate_alt_prior).t0 )
end

# ╔═╡ 44457c92-5f08-421f-b8ce-1cb3fee73c70
(; Δperiod = mode_result_to_named_tuple(map_estimate_ttv_model_jitter).period - mode_result_to_named_tuple(map_estimate_jitter).period, 
   Δt0 = mode_result_to_named_tuple(map_estimate_ttv_model_jitter).t0 -
		mode_result_to_named_tuple(map_estimate_jitter).t0 )

# ╔═╡ a62c28ab-61fc-4855-a29d-0eacad1534f1
days_to_minutes = 24*60

# ╔═╡ fcd3fe05-1425-424d-99a2-1c8c6be3bc4e
if @isdefined(map_estimate) 
	let
	pred_linear = predict_linear(df.n, mode_result_to_named_tuple(map_estimate))
	resid = df.t .- pred_linear
	plt = plot(legend=:topleft)
	scatter!(plt, df.t, resid.*days_to_minutes, label="Residuals to linear model")

	plot!(df.t, (predict_linear_plus_sinusoid(df.n,mode_result_to_named_tuple(map_estimate_ttv_model)).-pred_linear) .*days_to_minutes , label="TTV model for residuals")

	plot!(df.t, (predict_linear_plus_sinusoid(df.n,mode_result_to_named_tuple(map_estimate_ttv_model_jitter)).-pred_linear) .*days_to_minutes , label="TTV model for residuals w/ jitter")
	
	ylabel!(plt,"TTVs (min)")
	ylabel!(plt,"Time (d)")
	end
end

# ╔═╡ bc606b9a-0ae9-4517-9262-091ec59a8bf6
if false  # to create datafile from space delimited file
   lines = readlines("koi250.txt")
	df1 = DataFrame(map(l->(;n=parse(Int,l[9:12]), t_no_ttv=parse(Float64,l[14:24]), Δt=parse(Float64,l[26:35]), σₜ=parse(Float64,l[39:43])), lines[1:end-1]))
	df1.t = df1.t_no_ttv .+ df1.Δt./(24*60)

   CSV.write("koi250.01.csv", df1[!, [:n, :t, :σₜ, :t_no_ttv, :Δt]])
end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
PairPlots = "43a3c2be-4208-490b-832a-a21dcd55d7da"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.10.4"
ColorSchemes = "~3.18.0"
DataFrames = "~1.3.4"
Distributions = "~0.25.58"
ForwardDiff = "~0.10.29"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
MCMCChains = "~5.3.0"
Optim = "~1.7.0"
PDMats = "~0.11.10"
PairPlots = "~0.5.4"
Plots = "~1.29.0"
PlutoTeachingTools = "~0.2.3"
PlutoUI = "~0.7.38"
StatsPlots = "~0.14.34"
Turing = "~0.21.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "47aca4cf0dc430f20f68f6992dc4af0e4dc8ebee"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.0.0"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Setfield", "SparseArrays"]
git-tree-sha1 = "6320752437e9fbf49639a410017d862ad64415a5"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.5.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "345effa84030f273ee86fcdd706d8484ce9a1a3c"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.5"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "5d9e09a242d4cf222080398468244389c3428ed1"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.7"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "9ff1247be1e2aa2e740e84e8c18652bd9d55df22"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.3.8"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "e743af305716a527cdb3a67b31a33a7c3832c41f"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.5"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "81f0cb60dc994ca17f68d9fb7c942a5ae70d9ee4"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.8"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "cf6875678085aed97f52bfc493baaebeb6d40bcb"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.5"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "a83abdc57f892576bf1894d558e8a5c35505cdb1"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "de68815ccf15c7d3e3e3338f0bd3a8a0528f9b9f"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.33.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodeInfoTools]]
git-tree-sha1 = "91018794af6e76d2d42b96b25f5479bca52598f5"
uuid = "bc773b8a-8374-437a-b9f2-0e9785855863"
version = "0.3.5"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "1833bda4a027f4b2a1c984baddcf755d77266818"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.1.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a985dc37e357a3b22b260a5def99f3530fb415d3"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.2"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "8a6b49396a4058771c5c072239b2e0a76e2e898c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.58"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "a20d1374e896c72d2598feaf8e86b6d58a0c7d0a"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.39"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "Distributions", "LinearAlgebra", "MacroTools", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "5d1704965e4bf0c910693b09ece8163d75e28806"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.19.1"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "bed775e32c6f38a19c1dbe0298480798e6be455f"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.5.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "51c8f36c81badaa0e9ec405dcbabaf345ed18c84"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GPUArrays]]
deps = ["Adapt", "LLVM", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "c783e8883028bf26fb05ed4022c450ef44edd875"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.3.2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b316fd18f5bc025fedcb708332aecb3e13b9b453"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1e5490a51b4e9d07e8b04836f6008f46b48aaa87"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.3+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.IntervalSets]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "ad841eddfb05f6d9be0bff1fa48dcae32f134a2d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.6.2"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0f960b1404abb0b244c1ece579a0ec78d056a5d1"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.15"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "c8d47589611803a0f3b4813d9e267cd4e3dbcefb"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.11.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LRUCache]]
git-tree-sha1 = "d64a0aff6691612ab9fb0117b0995270871c5dfc"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.3.0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["CodeInfoTools", "FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "714193201458eb6281d8cf7fbe0e285d14516ba9"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.7.2"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "e9437ef53c3b29a838f4635e748bb38d29d11384"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.8"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "a9e3f4a3460b08dc75870811635b83afbd388ee8"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.3.0"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "058d08594e91ba1d98dcc3669f9421a76824aa95"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.3"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "74d7fb54c306af241c5f9d4816b735cb4051e125"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.4.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "f89de462a7bc3243f95834e75751d70b3a33e59d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.5"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded92de95031d4a8c61dfb6ba9adb6f1d8016ddd"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "e6c5f47ba51b734a4e264d7183b6750aec459fa0"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.11.1"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7a28efc8e34d5df89fc87343318b0a8add2c4021"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[deps.PairPlots]]
deps = ["Contour", "Latexify", "Measures", "NamedTupleTools", "PolygonOps", "Printf", "RecipesBase", "Requires", "StaticArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "3dd250135b0f48264f9a4e03d276fd96c8482ece"
uuid = "43a3c2be-4208-490b-832a-a21dcd55d7da"
version = "0.5.4"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "0e8bcc235ec8367a8e9648d48325ff00e4b0a545"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.5"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "d8be3432505c2febcea02f44e5f4396fae017503"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArrays", "LinearAlgebra", "RecipesBase", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "6b25d6ba6361ccba58be1cf9ab710e69f6bc96f8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.27.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "dad726963ecea2d8a81e26286f625aee09a91b7c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.4.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "30e3981751855e2340e9b524ab58c1ec85c36f33"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "8161f13168845aefff8dc193b22e3fcb4d8f91a9"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.31.5"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "3a2a99b067090deb096edecec1dc291c5b4b31cb"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "271a7fea12d319f23d55b785c51f6876aadb9ac0"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "43a316e07ae612c461fd874740aeef396c60f5f8"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.34"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0874c1b5de1b5529b776cfeca3ec0acfada97b1b"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.20"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "ca6c355677b0aed6b5dd5948824e77fee20f3ebf"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.2"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─1c651c38-d151-11ec-22a6-87180bd6546a
# ╟─d78db46d-1e20-4cd1-be53-81f635561b27
# ╟─a6966023-77c4-4744-9950-3cc27d0061e1
# ╟─20eb786a-f042-429f-a5de-f1af8f69b2fc
# ╠═d49b2d7a-fabe-4525-a11c-8e808960fc96
# ╟─90ae54f1-c49d-4d66-82eb-eb783c389e54
# ╟─efcabcc4-40ed-4492-b3e4-e190be248fce
# ╟─0873bbd9-f58b-4ec0-9506-15331f4a52fd
# ╟─c0e7a487-e8eb-4078-a473-e5fe64245f4c
# ╠═a77dd7a7-c9f7-4a93-9b85-207f88196650
# ╟─76904349-d28e-4f4f-8867-e5c4c73c3b19
# ╟─d88af6ab-f9c1-4aec-b335-106a3ab6ea45
# ╟─be82ab95-65b0-4de7-b406-14b796065087
# ╟─504f77c4-41db-4f70-b402-380fde67410d
# ╟─67ec0992-dd54-4172-b198-84bd61b7f48b
# ╟─e5aa5c21-4f7b-4011-bb7d-a3e73d3de0a5
# ╟─9e5c41c2-70c9-4b2b-84d0-dc223c818b31
# ╠═b8ab2460-7c59-4e87-8580-e7b49f0576aa
# ╟─3221c380-e04f-4822-ac5f-48add15aa757
# ╟─77799e4f-0aea-4fd4-9a85-a22eba5ab2d4
# ╟─81c18249-5951-48d9-a194-7f8e814cf8a3
# ╠═9926a73d-c8d0-4d83-9720-4313d82532bc
# ╟─3f1ee690-d9d0-4d39-ae3a-c7c42e7f0798
# ╟─86cb312e-84df-42a4-bad5-2d74abca2560
# ╠═c142f016-4b8d-4035-9dde-6c802c334546
# ╠═93ba5f00-a494-4c1c-8690-d50e3b398a7c
# ╟─4ffc0147-9477-4cfd-acc5-5ca7994caf20
# ╠═bf5e9688-993e-44fe-b6f0-6ab5daf2a4d8
# ╟─9355219a-8911-4ab9-ae68-5891743280b3
# ╠═24f0e74c-603d-4eb5-a8a7-1f602941a858
# ╟─3be23dcd-fd6d-40d0-a1ec-2eccc6222de3
# ╠═da63ae7c-1a40-4091-8aff-ef7de62c3bb7
# ╠═8317710d-50b5-4958-9a85-611133fd6ab8
# ╠═a9a7eca3-0562-42f4-9215-a877b6b2069a
# ╟─99bd8a04-9b1c-4bcc-a87b-01e0feeed341
# ╠═507395bd-a56c-4cdb-b7cd-182dcc9b1a34
# ╟─62f0b1e4-2885-433c-a13d-c87d6ae2fd9b
# ╠═8114804f-8ca8-43f0-b1d7-7c014270a77e
# ╠═308c16b2-0eb4-4df5-9436-ed6d6908f58c
# ╟─763a4a55-3f2f-46d6-98e7-f66327a2dc29
# ╠═bda05df7-0bc1-4025-88ac-109c0136618f
# ╟─b3363ef1-f046-49f3-b096-ad8cfb81e7f2
# ╟─325a5daf-e849-443c-95d8-630fe2d850c2
# ╟─d9d67f4f-01db-453a-bf69-5454a432b5fd
# ╠═b4fb0106-7981-4349-85b0-09e2bd6e48d8
# ╟─265123e6-f9f2-44f6-b1db-35f903ceaa9f
# ╟─63e72a21-24c3-410e-b8da-6c5abcfebb43
# ╟─fe5acfd9-fb85-4432-8f7e-90162fdeb5f7
# ╠═898cda4c-e193-4fdc-8139-7bb6ec858da0
# ╠═89fdecd6-57fd-4091-ac06-28edf04d0d01
# ╟─010b2e32-6bd4-47fa-a9c9-a0e14db92d8f
# ╠═5c4b0ec6-f684-4293-899a-339885af4e36
# ╠═1d526aee-7d67-4688-a163-7c3cffa3d6f3
# ╟─f93d7f0e-02b0-4935-9f5a-ab93ca28604b
# ╠═92ce947a-9f57-4efa-9b26-9d5e9efbd814
# ╟─2e433784-fd00-4fe9-a835-b37b7ec94b7c
# ╟─1993a166-9bea-4ded-9f12-5deb1327c133
# ╟─e8c0f93a-ebfb-47cb-bf88-5c2c66483683
# ╠═4ca9f2d6-d88d-49d5-8f1a-f8d464c7f7b9
# ╟─e746194b-63dc-444e-92cc-a2dbecdd6d2f
# ╠═2e15c59a-d990-453f-baa5-19128573df02
# ╠═eea61c7a-8dde-4a00-8b67-1b0adda052cf
# ╠═997dacbf-e0b1-4aac-977f-dd86180eb7dd
# ╠═09b185e3-a166-4168-81e4-2e9fb8a3132b
# ╠═ad96b064-e7a4-4b43-8368-56ae3e6059a2
# ╟─041aad0e-52e8-4ca2-80b1-24863d129ae4
# ╠═83f51ee0-eda6-4e14-83b1-7b76497e5b44
# ╟─f5048ccb-c549-483c-af8b-20482bac5cb6
# ╟─d1564b23-fdb7-499a-bab4-05363e6e7f53
# ╠═885ecd47-1093-4fe3-8e23-f5da27cce279
# ╠═fcd3fe05-1425-424d-99a2-1c8c6be3bc4e
# ╟─cd862907-3590-4b05-8715-797c2edae0bc
# ╠═969c184a-1a7f-4c3d-a33e-6bc8e61f8e96
# ╠═f379a950-18f2-439f-8c36-53ce8e461247
# ╠═70a61c2b-4292-417b-b306-ff09ad112f52
# ╠═efe787bc-5af9-4a45-9e4f-bb503ea83b22
# ╠═a91ef793-3c94-4f14-be04-d78d5f7106b5
# ╟─78a2895d-b059-469a-b3a7-1bacf7248757
# ╠═603a8abe-2e31-43f4-91ed-46857b65d089
# ╠═3fdf576d-abb6-4ef5-a1fb-1877fb122cc1
# ╠═f364ffc7-ce15-4bf1-ae80-f797c9d5f42b
# ╟─44457c92-5f08-421f-b8ce-1cb3fee73c70
# ╟─dcad1cad-f2c8-414e-9fed-e0de340e0603
# ╠═acb2f396-75e4-4475-93d8-54c72fd64189
# ╟─456f136d-1017-4a00-bc41-b8fcd4a6004b
# ╠═88be73c3-8049-489a-a265-254e29763be6
# ╟─a480bd7a-a4a9-43c7-81f5-77d15ec16289
# ╠═ab655734-a8b7-47db-9b73-4099c4b11dfc
# ╠═13a03add-4262-4bd0-9dc7-426a57b8e0e4
# ╠═a62c28ab-61fc-4855-a29d-0eacad1534f1
# ╟─bc606b9a-0ae9-4517-9262-091ec59a8bf6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
