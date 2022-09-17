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

# ╔═╡ 6fd32120-4df1-4f2d-bb6f-c348a6999ad5
begin
	using PlutoUI, PlutoTeachingTools, HypertextLiteral
	using CSV, DataFrames
	using Turing,  Optim, MCMCChains
	using Plots, LaTeXStrings,StatsPlots
	using DataFrames
	using LinearAlgebra, PDMats
	using Statistics: mean, std
	using Distributions
	using Random
	Random.seed!(123)
end;

# ╔═╡ 10709dbc-c38b-4f15-8ea8-772db2acfbb3
md"""# Model Building III: Bayesian Inference withh Non-Linear Models
**Astro 497, Lab 5, Exercise 2**
"""

# ╔═╡ 47a0c0de-7d75-4891-b671-c083578a004d
TableOfContents()

# ╔═╡ 73021015-4357-4196-868c-e9564de02ede
md"""
## Overview
In this lab, we'll develop a model for analyzing observations from a radial velocity survey.  
As before, we'll start by describing a physical model and a measurement model. 
Together, those two define a likelihood.  
Since we'll work in a Bayesian framework, we'll need to specify priors.   
Then, we'll using [Turing.jl](https://turing.ml/stable/) to implement that model and perform posterior sampling.
Finally, you'll make some modest changes to the model, perform posterior sampling with your revised model, and compare the results. 
"""

# ╔═╡ 13d4e147-d12a-43e5-883d-b037a8e3b433
md"""
## Likelihood 
In either Frequentist or Bayesian statistics, we use a [likelihood](https://en.wikipedia.org/wiki/Likelihood_function) function (the probability distribution for a given set of observations for given values of the model parameters).
Rather than writing an explicit likelihood ourselves, we'll specify a physical model and a measurement model, and let Turing figure out how to compute the likelihood.
"""

# ╔═╡ 0574750f-04d1-4677-b52d-ac3aa68c6eee
md"""
### Physical Model
We aim to infer the orbital properties of an exoplanet orbiting a star based on radial velocity measurements.  
We will approximate the motion of the star and planet as a Keplerian orbit.  In this approximation, the radial velocity perturbation ($Δrv$) due to the planet ($b$) is given by 
```math
\mathrm{Δrv}_b(t) = \frac{K}{\sqrt{1-e^2}} \left[ \cos(ω+T(t,e)) + e \cos(ω) \right],
```
where the orbital eccentricity ($e$), the arguement of pericenter ($\omega$) which specifies the direction of the pericenter, and $T(t,e)$ the [true anomaly](https://en.wikipedia.org/wiki/True_anomaly) at time $t$ given the eccentricity $e$.
$K$ is a function of the planet and star masses, the orbital period and the sine of the orbital inclination relative to the sky plane.
"""

# ╔═╡ f6512ab6-2783-43c1-81a5-9c05c539b749
md"""
The true anomaly ($T$) is related to the eccentric anomaly ($E$) by
```math
\tan\left(\frac{T}{2}\right) = \sqrt{\frac{1+e}{1-e}} \tan\left(\frac{E}{2}\right).
```
"""

# ╔═╡ 1492e4af-440d-4926-a4b5-b33da77dbee2
function calc_true_anom(ecc_anom::Real, e::Real)
	true_anom = 2*atan(sqrt((1+e)/(1-e))*tan(ecc_anom/2))
end

# ╔═╡ 9f7196d2-83c4-4e6a-baa7-b8a17f51b6e3
md"""
The mean anomaly increases linearly in time.
```math
M(t) = 2π(t-t_0)/P + M_0.
```
Here $M_0$ is the mean anomaly at the epoch  $t_0$.

High-resolution spectroscopy allows for precision *relative* radial velocity measurements.  Due to the nature of the measurement process, there is an arbitrary velocity offset ($C$), a nuisance parameter.
```math
\mathrm{rv(t)} = \mathrm{Δrv}_b(t) + C.
```
"""

# ╔═╡ a85268d5-40a6-4654-a4be-cba380e97d35
md"### Measurement model"

# ╔═╡ cfe8d6ad-2125-4587-af70-875e7c4c4844
md"""
In order to perform inference on the parameters of the physical model, we must specify a measurement model.  
We will assume that each observation ($rv_i$) follows a normal distribution centered on the true radial velocity ($\mathrm{rv}(t_i)$) at time $t_i$ and that the measurement errors are independent of each other.
```math
L(θ) \sim \prod_{i=1}^{N_{\mathrm{obs}}} N_{\theta}( \mathrm{rv}_i - \mathrm{rv}_{\mathrm{true}}(t_i | \theta), \sigma_{\mathrm{eff},i}^2).
```
Above, $N_x(\mu,\sigma^2)$ indicates a normal probability distribution for $x$ with mean $\mu$ and variance $\sigma^2$.  
We assume that the variance for each measurement ($\sigma_{\mathrm{eff},i}^2$) is given by 
```math
\sigma_{\mathrm{eff},i}^2 = \sigma_i^2 + \sigma_{\mathrm{jitter}}^2, 
```
where $\sigma_i$ is the estimated measurement uncertainty for the $i$th measurement
and $\sigma_{\mathrm{jitter}}$ parameterizes any additional ``noise'' source (e.g., stellar variability, undetected planets, additional noise from instrument or data reduction pipeline).  
"""

# ╔═╡ b9b14b8d-f843-48f8-afe5-4d8f3008460a
md"""
## Priors
When performing Bayesian inference, we must specify a [prior distribution](https://en.wikipedia.org/wiki/Prior_probability) for the model parameters (θ).  
Often we assume that the prior for the model parameter is a product of univariate priors for individual parameters.		
For some variables we'll use common distributions provided by the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package.
"""

# ╔═╡ 9ffab6aa-3e9c-4daf-91f5-af3971528341
md"""
#### Angles 
One strategy for picking priors is to choose a probability distribution that is non-informative given some constraints.  
This is often known as the [Jeffreys prior](https://en.wikipedia.org/wiki/Jeffreys_prior).
For variables that specify a location and that are confined to a finite range, the Jeffreys prior would be Uniform over values within the allowed bounds.   
For example, for the two angles $ω$ and $M_0$, we'll adopt uniform priors over the full range $[0,2\pi)$.
This can be understood physically, since there's no reason why planetary orbits would prefer pointing in one direction or another.  
Similarly, there's no reason that whatever time we define as $t=0$ would be special.  
(Actually, we'll parameterize the model in terms of $\omega$ and and $M_0-\omega$ rather than $\omega$ and $M_0$, since $M_0-\omega$ usually have a lower correlation with each other than $\omega$ than $M_0$ do.)

Since the uniform distribution is standard, it will be easy to write these directly in our model using
```julia
ω ~ Uniform(0, 2π)           # arguement of pericenter
M0_minus_ω ~ Uniform(0,2π)   # mean anomaly at t=0 minus ω
```
"""

# ╔═╡ 9a7302cd-3b51-4b1e-b932-6b5785998a8a
md"""
#### Velocity Offset
Next, we'll consider the velocity offset $C$ between the velocity measured in each observation and some reference epoch.  
The main effect causing the velocity offset is the motion of the Earth around the Sun.  
If the target star lies in the plane of the Earth-Sun orbit, then the Earth's orbital motion is ~30km/s.  
If the target star is nearly perpendicular to the Earth's orbit, then the line-of-sight orbital motion goes to zero.  
Often astronomers use a uniform prior.  However, this would require specifying hard upper and lower limits.  
An alternative is to specify a Normal distribution with a large variance.  
That way there's at least some small non-zero probability for any real value, but effectively the prior is flag for the small velocity offsets that we expect to be typical.   So we'll use
```julia
C ~ Normal(0,30_000.0)         # velocity offset
```
"""

# ╔═╡ 54467bb4-6901-4934-8c11-7f32012406c8
md"""
#### Eccentricity
For the eccentricity prior, often astronomers adopt a Uniform prior, using logic similar to above.  
However, physically, it would be suprising in there were just as many planets with $e$ in [0.95,1) as in [0,0.05). 
Astronomers have proposed a variety of alternatives, and none is perfect for all circumstances.  
Here, we'll adopt a [Rayleigh distribution](https://en.wikipedia.org/wiki/Rayleigh_distribution).
This is motivated by the idea that planets form on nearly circular orbits and small gravitational perturbation from other planets (or protoplanets) cause a random walk of their epicyclic motions.  
The choice of the scale parameter is motivated by previous observations of other giant exoplanets.   
The Rayleigh distribution is defined on [0,∞), but orbits with e≥1 will be unbound.  And orbits with e≃1 would result in the planet passing through the star at pericenter.   So we'll truncate the Rayleigh distribution to be confined to be less than unity.  
"""

# ╔═╡ b289d8f7-b138-43e7-98a1-d971089d7f72
prior_e = Truncated(Rayleigh(0.3),0.0,0.999);

# ╔═╡ 21612349-dbc5-4bdd-8036-12900a005b21
md"""
#### Period & Amplitude 
Each of the period and $K$ are non-negative scale parameters. 
The non-informative Jeffreys prior would be $p(x) ∝ 1/x$ for $x>0$.  
However, this diverges (at both $0$ and $\infty$), making it an *improper prior*.  
While some Bayesian computations can be done using improper priors, it's generally a good habit to use proper priors when practical.  
In our case, one could motivate a hard limit on the minimum orbital period based on the Roche limit. 
But there's no hard limit to how low mass a planet could be or how little extra noise might be in our dataset.  
So rather than truncating the prior at small values, we'll have the prior flatten below some scale.  
We'll truncate the distribution at the upper end, based on the largest plausible values.  

For each of $P$ and $K$, we'll adopt a modified Jeffreys prior for a scale parameter
```math
p(x) ∝ \frac{1}{1+\frac{x}{x_0}}, \qquad 0 \le x \le x_{\max}
```
as suggested in [Ford & Gregory (2007)](https://ui.adsabs.harvard.edu/#abs/2007ASPC..371..189F/abstract). 
In the case of $K$ this makes sense, since the quality of our dataset wouldn't be good enough to discover a planet with a $K$ significantly less than our single measurement precision (of order a few m/s).
For $P$, we can think of it as saying that we won't assign zero prior probability to planets in the middle of spiraling into their star, but we also won't let the prior probability continue to increase below some minimum period (which we'll take to be 1 day).  
"""

# ╔═╡ a1a7d4b7-8c02-4e74-a158-3e05b4da63fb
md"""
#### Jitter parameter
In principle, the Jitter parameter, $\sigma_j$, could take on any non-negative value. 
 In practice, we expect that effects like stellar variability and unmodeled instrument noise are likely to be roughly ~1-5 m/s.  
But an undetected planet could cause much larger scatter and we don't want to rule out that possibility.
So we'll adopt a log Normal distribution with a broad peak around 3m/s.  
"""

# ╔═╡ 42589e2f-3b4d-4764-8f96-00614ed144e7
prior_jitter = LogNormal(log(3.0),1.0);

# ╔═╡ a2140dbe-8736-4ed9-ae6f-b1b0c7df3bc9
md"""
## Ingest & Validate Data
"""

# ╔═╡ 22719976-86f5-43d3-b890-d3520f9916d2
md"""
We will read in a DataFrame containing radial observations of the star 16 Cygni B from Keck Observatory.  (Data reduction provided by [Butler et al. 2017](https://ui.adsabs.harvard.edu/link_gateway/2017AJ....153..208B/doi:10.3847/1538-3881/aa66ca) and the [Earthbound Planet Search](https://ebps.carnegiescience.edu/) project.)  The star hosts a giant planet on an eccentric orbit ([Cochran et al. 1097](https://doi.org/10.1086/304245)).  In this exercise, we will construct a statistical model for this dataset.
"""

# ╔═╡ e50bdd14-d855-4043-bbab-f6526a972e31
begin
	fn = joinpath("data","16cygb.txt")
	df = CSV.read(fn,DataFrame,header=[:Target,:bjd,:rv,:σ_rv,:col5,:col6,:col7,:col8],skipto=100,delim=' ',ignorerepeated=true)
end;

# ╔═╡ 87dd11ad-c29e-4a5e-90b1-289863eedd57
md"""
It's often useful to do a quick plot to make sure there aren't any suprises (e.g., read wrong column into wrong variable, unexpected units, outliers, etc.).
"""

# ╔═╡ ddac2663-fd90-4c60-acaa-da2560367706
let 
	plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd],df[!,:rv],yerr=df[!,:σ_rv])
	xlabel!(plt,"Time (BJD)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Keck Observations of 16 Cyb B")
end

# ╔═╡ b9efe114-717e-46c8-8225-a4ab4d3df439
md"""
In order to numerical errors due to floating arithmetic small, we'll adopt a reference epoch near many of our observations.   
"""

# ╔═╡ 2844cd3a-9ed1-47da-ab59-ea33575b4991
bjd_ref = 2456200.0;

# ╔═╡ f28d4bc8-53e7-45f9-8126-7339a6f54732
md"# Bayesian Inference using PPL"

# ╔═╡ 342a0868-4f28-444d-b858-91ccdc655561
md"## Define model"

# ╔═╡ 56aa2fd5-fb34-495c-9b9f-0ce0bbbd6b1b
md"""
We'll define a probabilistic model use Turing's `@model` macro applied to a julia function.  Our model function takes the observed data (in this case the observation times, radial velocities and the estimated measurement uncertainties) as function arguements.  
In defining a probabilistic model, we specify the distribution that each random variable is to be drawn from using the `~` symbol.  Inside the model, we can specify transformations, whether simple arithmetic and or complex functions calls based on both random and concrete variables.   
"""

# ╔═╡ 228bb255-319c-4e80-95b3-8bf333be29e4
md"""
Remember that our model is written as a function of the observational data.  
Therefore, we will specify a posterior probability distribution for a given set of observational data.  
"""

# ╔═╡ 53240de7-f147-485c-b113-ff20aaa1227b
md"""
## Initial attempt to fit model
We could attempt to find the maximum a priori (MAP) parameter values for this model model using the same approach as in the previous lab fitting linear (or nearly linear) model.  Let's try.
"""

# ╔═╡ 60b2a73a-e0dc-4e9c-a078-b3fd34f0def7
md"""
**Q1a:** Try rerunning the cell a few times (each time it will start from a different initial guess).  Based on the parameter values returned, have we found a good model?  If not, what tips you off that there's a problem?  
"""

# ╔═╡ 795011db-ec97-4f2b-907d-14fa35094b54
response_1a = missing

# ╔═╡ 1e143b7b-3598-4048-9133-2110944b3a67
if ismissing(response_1a) still_missing() end

# ╔═╡ 5ab4a787-fe9b-4b2c-b14c-273e0205259d
md"""
## Initial Attempt to Sample from the Posterior
"""

# ╔═╡ 5f20e01d-3489-4828-9e4e-119095e9c29c
md""" 
Since our model includes non-linear functions, the posterior distribution for the model parameters can not be computed analytically.  Fortunately, there are sophisticated algorithms for sampling from a probability distribution (e.g., [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo); MCMC).  Soon, we will compute a few posterior samples and investigate the results.

The observation that a simple optimization strategy didn't work well above, is a bit of a warning sign.  There's a chance that posterior sampling might work better, but we'll proceed with caution.  
"""

# ╔═╡ dfdf843d-98ce-40a1-bd0b-0a11f1cdb5f9
md"""
The next calculation can take seconds to minutes to run, so I've provide boxes to set some parameters below.  The default number of steps per Markov chain is much smaller than you'd want to use for a real scientific study, but is likely enough to illustrate the points for this lab.  Near the end of this exercise (i.e., after you've made a good initial guess for the orbital period below), feel free to dial it up and  wait a few to several minutes to see the improved results for the subsequent calculations.
"""

# ╔═╡ 87a2281b-8827-4eca-89a6-c7d77aa7a00f
md"Number of steps per Markov chain  $(@bind num_steps_per_chain NumberField(100:100:10_000; default=300)) "

# ╔═╡ 9f27ff9f-c2c4-4bf4-b4cd-bac713de0abe
md"Number of Markov chains  $(@bind num_chains NumberField(1:10; default=4)) "

# ╔═╡ 358ab0f4-3423-4728-adfa-44126744ae18
md"""
In the cell above, we called Turing's `sample` function applied to the probability distribution given by `posterior_1`, specified that it should use the [No U-Turn Sampler (NUTS)](https://arxiv.org/abs/1111.4246), asked for the calculation to be parallelized using multiple threads, and specified the number and length of Markov chains to be computed.
"""

# ╔═╡ c1fa4faa-f10c-4ed0-9539-20f5fcde8763
md"### Inspecting the Markov chains"

# ╔═╡ 00c4e783-841c-4d6a-a53d-34a88bbe1f7a
md"""
In the above calculations, we drew the initial model parameters from the prior probability distrirbution.  Sometimes those are very far from the true global mode.  For a simple model, it's possible that all the chains will (eventually) find the global mode.  However, the different Markov chains might have  gotten "stuck" in different local maxima of the posterior density, depending on where each started.  We can visualize how the Markov chains progressed by looking at a trace plot for any of the model parameters.  Below, we show a large trace plot for period and smaller trace plots for four other parameters.
"""

# ╔═╡ f7046ee4-9eb7-4324-886e-5a3650b58db7
tip(md"""
Since we know the initial state of each Markov chain are strongly influenced by our initial guess, we usually discard the first portion of each Markov chain.  Normally, Turing does this automatically.  Above, we explicitly passed the optional arguement 'discard_initial=0', so that we could make the above plot and see where each chain started.
""")

# ╔═╡ bb2ff977-2d12-4baa-802c-097a4138a24b
md"""
The [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl) package provides a `describe` function that provides a simple summary of Markov chain results.  
"""

# ╔═╡ 1196e7e6-4144-47e8-97cd-6f93530e11e5
md"""
*In theory*, each Markov chains is guaranteed to converge to the same posterior distribution, *if* we were to run it infinitely long.  In practice, we can't run things infinitely long. So we must be careful to check for any indications that the Markov chains might not have converged.  The above table has some statistics that can be useful for recognizing signs of non-convergence.  But since this isn't a statistics class, we'll just use one simple qualitative test.  Since we've computed several Markov chains (potentially in parallel to save our time), we can compare the results.  If they don't agree, then they can't all have converged to the same distribution.
"""

# ╔═╡ 55a3378a-da18-4310-8272-cf11298a118d
md"""
We can inspect the results of each of the Markov chains separately, by taking a 2-D "slice", where we fixed the value of the third axis to `chain_id` set by the slider below.
"""

# ╔═╡ ac237181-b6d2-4968-b6e3-146ba41bcc18
md"""
Below, we'll visualize the results for one important parameter by ploting an estimate of the marginal density for the orbital period from each of the Markov chains we've computed.  
"""

# ╔═╡ a8e547c5-c2b5-4058-a1e2-35397d7e8112
md"**Q2a:**  Based on the summary statistics and/or estimated marginal posterior density for the orbital period, did the different Markov chains provide qualitatively similar results for the marginal distribution of the orbital period?"

# ╔═╡ 80d076a8-7fb1-4dc7-8e3b-7579db30b038
response_2a = missing

# ╔═╡ 3759ec08-14f4-4026-916b-6a0bbc818ca7
if ismissing(response_2a) still_missing() end

# ╔═╡ 2fa23729-007e-4dcb-b6f9-67407929c53a
tip(md"""
Since it's a random algorithm, I can't be 100% sure what results you'll get.  My guess is that if you ran several Markov chains for the suggested model and dataset, then some will likely have resulted in substantially different marginal density estimates.
""")

# ╔═╡ 52f4f8bd-41a9-446f-87e7-cdc1e9c178f8
md"### Visualize the model predictions"

# ╔═╡ bf5b40c5-399e-40a0-8c27-335f72015e58
md"Another good strategy to help check whether one is happy with the results of a statistic model is to compare the predictions of the model to the observational data.  Below, we'll draw several random samples from the posterior samples we computed and plot the model RV curve predicted by each set of model parameters." 

# ╔═╡ 98bc7209-a6a1-4c06-8e6f-de5b421b7b8c
t_plt = range(first(df.bjd)-bjd_ref-10, stop=last(df.bjd)-bjd_ref+10, length=400);

# ╔═╡ 207a0f69-30ce-4107-a21d-ba9f6c5a5039
 n_draws = 10;

# ╔═╡ 93115337-ec60-43b9-a0ef-1d3ffc1fe5ff
@bind draw_new_samples_from_posterior_with_init_from_prior Button("Draw new samples")

# ╔═╡ 0d83ebc9-ecb3-4a9d-869b-351b28d5e962
md"**Q2b:**  Are any/some/most/all of the predicted RV curves good fits to the data?"

# ╔═╡ cc2dbf2f-27a4-4cd6-81e5-4a9f5a3bc488
response_2b = missing

# ╔═╡ f00539c9-ee40-4128-928c-460fd439fa87
if ismissing(response_2b) still_missing() end

# ╔═╡ b787d7ef-bda6-470d-93b8-954227dec24b
md"# Global search with an approximate model"

# ╔═╡ aaf24ba9-48dd-49ff-b760-2b806f7cf5ba
md"""
It turns out that the posterior distribution that we're trying to sample from is *highly* multimodal.  We can get a sense for this by considering a *periodogram*.  For our purposes, this is simply an indication of the goodness of the best-fit model for each of many possible periods.  To make the calculations faster, we've assumed that the orbits are circular, allowing us to make use of generalized linear least squares fitting which is much faster than fitting a non-linear model.  This allows us to quickly explore the parameter space and identify regions worthy of more detailed investigation.  
"""

# ╔═╡ e62067ce-dbf6-4c50-94e3-fce0b8cbef12
md"""
We can see that there are multiple orbital periods that might be worth exploring, but they are separated by deep valleys, where models would not fit well at all.  Strictly speaking, the ideal Markov chain would repeatedly transition between orbital solutions in each of the posterior modes.  In practice, most MCMC algorithms are prone to get stuck in one mode for difficult target densities like this one.   Therefore, we will compute new Markov chains using the same physical and statistical model, but starting from initial guesses near the posterior mode that we want to explore.  In practice, it would be even better to try starting Markov chains near each of the potentially good modes to see how similar/different the resulting models are. 
"""

# ╔═╡ dd918d00-fbe7-4bba-ad6a-e77ea0acb43a
md"## Initial guess for model parameters"

# ╔═╡ 9c82e944-d5f2-4d81-9b33-7e618421b715
md"""
Below, I've provided a pretty good guess below that is likely to give good results.  After you've completed through the lab, you're encouraged to come back to this part and see what happens when you try initializing the Markov chains with different initial states.  For example, you could try seeing what happens if you start near another periodogram peak.
"""

# ╔═╡ b76b3b7f-c5f3-4aec-99b5-799479c124d6
begin
	P_guess = 710.0         # d
	K_guess = 40.9          # m/s
	e_guess = 0.70
	ω_guess = 1.35          # rad
	M0_minus_ω_guess = 4.35 # rad
	C_guess = -2.3          # m/s
	jitter_guess = 3.0      # m/s
	param_guess = (;P=P_guess, K=K_guess, e=e_guess, ω=ω_guess, M0_minus_ω=M0_minus_ω_guess, C=C_guess, σ_j=jitter_guess)
end

# ╔═╡ 0085a6ce-334a-46a7-a12d-ee7b61c59f23
md"""
We can try running a standard optimizer from this initial guess.
"""

# ╔═╡ 3459436a-bfeb-45d4-950e-68fd55af76d7
md"## Visualize model predictions"

# ╔═╡ 47e61e3a-466d-40f9-892e-5691eb6a2393
md"""
Next, we'll try computing a new set of Markov chains using our guess above to initialize the Markov chains.
"""

# ╔═╡ ba4e0dc6-5008-4ed8-8eba-3902820cf241
md"# Sampling from the posterior distribution with an initial guess"

# ╔═╡ a247788c-8072-4cc4-8c38-68d0a3445e83
md"""
I'm ready to compute posterior sample with new guess: $(@bind go_sample_posterior1 CheckBox(default=true))

(Uncheck box above if you want to inspect the predicted RV curve using several different sets of orbital parameters before you compute a new set of Markov chains.)
"""

# ╔═╡ 565d68a4-24cd-479a-82ab-21d64a6a01f6
md"### Inspecting the new Markov chains"

# ╔═╡ db2ba5d1-3bd3-4208-9805-9c3fab259377
md"""
Again, we'll check the summary of posterior samples for the  model parameters for the group of Markov chains and for each chain individually.
"""

# ╔═╡ c9433de9-4f48-4bc6-846c-3d684ae6adee
md"We can also compare the marginal posterior density estimated from each Markov chain separately for each of the model parameters." 

# ╔═╡ 7f8394ee-e6a2-4e4f-84b2-10a043b3da35
md"**Q2c:**  Based on the summary statistics and estimated marginal posterior densities, did the different Markov chains provide qualitatively similar results for the marginal distributions of the orbital period and other model parameters?"

# ╔═╡ f2ff4d87-db0a-4418-a548-a9f3f04f93cd
response_2c = missing

# ╔═╡ 42fe9282-6a37-45f5-a833-d2f6eb0518fe
if ismissing(response_2c) still_missing() end

# ╔═╡ b38f3767-cd14-497c-9576-22764c53a48d
protip(md"Assessing the performance of Markov chain Monte Carlo algorithm could easily be the topic of a whole lesson.  We've only scratched the surface here.  You can find [slides](https://astrostatistics.psu.edu/su18/18Lectures/AstroStats2018ConvergenceDiagnostics-MCMCv1.pdf) from a lecture on this topic for the [Penn State Center for Astrostatistics](https://astrostatistics.psu.edu) summer school.")  

# ╔═╡ 35562045-abba-4f36-be92-c41f71591b1a
md"### Visualize predictions of new Markov chains"

# ╔═╡ e9530014-1186-4897-b875-d9980e0c3ace
md"Below, we'll draw several random samples from the posterior samples we computed initializing the Markov chains with our guess at the model parameters and plot the model RV curve predicted by each set of model parameters." 

# ╔═╡ 9dc13414-8266-4f4d-94dd-f11c0726c539
@bind draw_new_samples_from_posterior_with_guess Button("Draw new samples")

# ╔═╡ 95844ef7-8730-469f-b69b-d7bbe5fc2607
md"**Q2d:**  Are any/some/most/all of the predicted RV curves good fits to the data? "

# ╔═╡ 3699772e-27d4-402e-a669-00f5b22f2ed5
response_2d = missing

# ╔═╡ 9eccb683-d637-4ab9-8af7-70c24a7d8478
!ismissing(response_2d) || still_missing()

# ╔═╡ 9a2952be-89d8-41ef-85ce-310ed90bd0d1
md"# Generalize the Model"

# ╔═╡ c7e9649d-496d-44c2-874c-5f51e311b21d
md"""
One of the benefits of probabilistic programming languages is that they make it relatively easy to compare results using different models.
The starter code below defines a second model identical to the first.  Now, it's your turn to modify the model to explore how robust the planet's orbital parameters are.  

The star 16 Cygni B is part of a wide binary star system.  The gravitational pull of 16 Cygni A is expected to cause an acceleration on the entire 16 Cygni B planetary system.  Since the separation of the two stars is large, the orbital period is very long and the radial velocity perturbation due to the star A over the short time span of observations can be approximated as a constant acceleration,  
```math
\Delta~\mathrm{rv}_A = a\times t,
```
and the observations can be modeled as the linear sum of the perturbations due to the planet and the perturbations due to the star,
```math
\mathrm{rv(t)} = \mathrm{Δrv}_b(t) + \mathrm{Δrv}_A(t) + C.
```

**Q3a:** Update the model below to include an extra model parameter ($a$) for the acceleration due to star A and to include that term in the true velocities.
You'll need to choose a reasonable prior distribution for $a$.    
"""

# ╔═╡ 5084b413-1211-490a-89d3-1cc782f1741e
md"## Sampling from the new model"

# ╔═╡ c6939875-0efc-4f1d-a2d3-6b46484328a5
md"""
Since we have a new statistical model, we'll need to define a new posterior based on the new statistical model and our dataset.
"""

# ╔═╡ 43177135-f859-423d-b70c-e755fdd06765
md"""
Since the new model includes an extra variable, we'll need to update our initial guess to include the new acceleration term.  We'll choose a small, but non-zero value.  (Using a to exactly zero could make it hard for the sampling algorithm to find an appropriate scale.)  
"""

# ╔═╡ c33d9e9d-bf00-4fa4-9f90-33415385507e
param_guess_with_acc = merge(param_guess, (;a = 0.0) )

# ╔═╡ 20393a82-6fb1-4583-b7b7-8d1cda43bd47
md"""
We're you're ready to start sampling from the new posterior, check the box below.

Ready to sample from new posterior using your new model? $(@bind go_sample_posterior2 CheckBox(default=false))
"""

# ╔═╡ 13c56edd-3c1e-496e-8116-fb158dd0f133
md"## Inspect Markov chains for generalized model"

# ╔═╡ c2b7dfcc-6238-4e9a-a2b8-efcc05c17361
md"""
Let's inspect summary statistics and marginal distributions as we did for the previous model.
"""

# ╔═╡ 21fdeaff-0c91-481c-bd8e-1dba27e275a6
md"""
**Q3b:**  Based on the the above summary statistics and estimated marginal posterior densities from each of the Markov chains, do you see any reason to be suspicious of the results from the new analysis using the model with a long-term acceleration?  If so, explain
"""

# ╔═╡ a4be55ab-3e8e-423c-b3bc-3f3b88c5d2b7
response_3b = missing

# ╔═╡ 5efb1eac-c1fe-417f-828a-3cfb8978da40
!ismissing(response_3b) || still_missing()

# ╔═╡ bcedc85f-8bf2-49d4-a60a-d6020450fd76
md"## Visualize predictions of generalized model"

# ╔═╡ d751f93c-9ef4-41ad-967b-7ccde6e40afd
@bind draw_new_samples_from_posterior_with_acc Button("Draw new samples")

# ╔═╡ 5171f2f0-e60c-4038-829f-9baf2d5f178e
md"To see if the model better describes our data, we can inspect the histogram of residuals between the observations and the model predictions for each of the models.  "

# ╔═╡ d34e6593-d847-4728-addb-4d2ebe32fdc0
md"""
**Q3c:**  Based on comparing the predictions and observations, do you see any reason to be suspicious of the results from the new analysis using the model with a long-term acceleration?  If so, explain.
"""

# ╔═╡ 47be22e6-4d81-4849-a4f8-08f4d4537829
response_3c = missing

# ╔═╡ 68134fbb-5e31-4abf-babe-1508d7c692df
!ismissing(response_3c) || still_missing()

# ╔═╡ 083e9418-9b64-46d0-8da4-3396fb958862
md"Standardize Residuals? $(@bind standardize_histo CheckBox(default=false))"

# ╔═╡ 8180b43d-81aa-4be0-bdf1-ac93f734331c
md"# Compare results using two different models"

# ╔═╡ d9cd7102-490d-4f35-a254-816c069d3810
md"""
Finally, we'll compare our estimates of the marginal posterior distributions computed using the two models.   
"""

# ╔═╡ 9fd50ada-702f-4ca4-aab2-abfa0f4f597c
md"""
**Q3d:**  Did the inferences for the orbital period or velocity amplitude change significantly depending on which model was assumed?  
"""

# ╔═╡ b3f9c7b7-5ed5-47d7-811c-6f4a313de24b
response_3d = missing

# ╔═╡ 461416e4-690f-4ebd-9f07-3e34962c8693
!ismissing(response_3d) || still_missing()

# ╔═╡ e5d0b1cc-ea7b-42a4-bfcd-684337b0f98b
md"""
**Q3e:**  Based on the results above how accurately can we measure the masss of the planet 16 Cyb b?
"""

# ╔═╡ 5e5d4560-fa1e-48f6-abe4-3b1221d44609
response_3e = missing

# ╔═╡ dd40a3cf-76d3-4eb9-8027-274a065c762c
!ismissing(response_3e) || still_missing()

# ╔═╡ 940f4f42-7bc3-48d4-b9f4-22f9b94c345d
md"""
# Helper Code
"""

# ╔═╡ 40923752-9215-45c9-a444-5a519b64df97
ChooseDisplayMode()

# ╔═╡ 83485386-90fd-4b2d-bdad-070835d8fb44
md"""
## Solving Kepler's Equation
"""

# ╔═╡ b01b2df0-9cfd-45c8-ba35-cd9ec018af6a
"""
   ecc_anom_init_guess_danby(mean_anomaly, eccentricity)

Returns initial guess for the eccentric anomaly for use by itterative solvers of Kepler's equation for bound orbits.  

Based on "The Solution of Kepler's Equations - Part Three"
Danby, J. M. A. (1987) Journal: Celestial Mechanics, Volume 40, Issue 3-4, pp. 303-312 (1987CeMec..40..303D)
"""
function ecc_anom_init_guess_danby(M::Real, ecc::Real)
	@assert -2π<= M <= 2π
	@assert 0 <= ecc <= 1.0
    if  M < zero(M)
		M += 2π
	end
    E = (M<π) ? M + 0.85*ecc : M - 0.85*ecc
end;

# ╔═╡ f104183b-ea56-45c3-987e-94e42d687143
"""
   update_ecc_anom_laguerre(eccentric_anomaly_guess, mean_anomaly, eccentricity)

Update the current guess for solution to Kepler's equation
  
Based on "An Improved Algorithm due to Laguerre for the Solution of Kepler's Equation"
   Conway, B. A.  (1986) Celestial Mechanics, Volume 39, Issue 2, pp.199-211 (1986CeMec..39..199C)
"""
function update_ecc_anom_laguerre(E::Real, M::Real, ecc::Real)
  (es, ec) = ecc .* sincos(E)  # Combining sin and cos provides a speed benefit
  F = (E-es)-M
  Fp = one(M)-ec
  Fpp = es
  n = 5
  root = sqrt(abs((n-1)*((n-1)*Fp*Fp-n*F*Fpp)))
  denom = Fp>zero(E) ? Fp+root : Fp-root
  return E-n*F/denom
end;

# ╔═╡ 097fab0c-edfc-4d3a-abb2-4285b026e3f2


# ╔═╡ 9c50e9eb-39a0-441a-b03f-6358caa2d0e9
begin
	"""
	   calc_ecc_anom( mean_anomaly, eccentricity )
	   calc_ecc_anom( param::Vector )
	
	Estimates eccentric anomaly for given 'mean_anomaly' and 'eccentricity'.
	If passed a parameter vector, param[1] = mean_anomaly and param[2] = eccentricity. 
	
	Optional parameter `tol` specifies tolerance (default 1e-8)
	"""
	function calc_ecc_anom end
	
	function calc_ecc_anom(mean_anom::Real, ecc::Real; tol::Real = 1.0e-8)
	  	if !(0 <= ecc <= 1.0)
			println("mean_anom = ",mean_anom,"  ecc = ",ecc)
		end
		@assert 0 <= ecc <= 1.0
		@assert 1e-16 <= tol < 1
	  	M = rem2pi(mean_anom,RoundNearest)
	    E = ecc_anom_init_guess_danby(M,ecc)
		local E_old
	    max_its_laguerre = 200
	    for i in 1:max_its_laguerre
	       E_old = E
	       E = update_ecc_anom_laguerre(E_old, M, ecc)
	       if abs(E-E_old) < tol break end
	    end
	    return E
	end
	
	function calc_ecc_anom(param::Vector; tol::Real = 1.0e-8)
		@assert length(param) == 2
		calc_ecc_anom(param[1], param[2], tol=tol)
	end;

	calc_ecc_anom_url = "#" * (PlutoRunner.currently_running_cell_id[] |> string)
end;

# ╔═╡ 0ad398fb-9c7e-467d-a932-75db70cd2e86
begin 
	# First, we provide a single docstring for all versions of the function (with the same name).
	""" Calculate RV from t, P, K, e, ω and M0	"""
	function calc_rv_keplerian end   

	function calc_rv_keplerian(t, P,K,e,ω,M0) 
		mean_anom = t*2π/P-M0
		ecc_anom = calc_ecc_anom(mean_anom,e)
		true_anom = calc_true_anom(ecc_anom,e)
		rv = K/sqrt((1-e)*(1+e))*(cos(ω+true_anom)+e*cos(ω))
	end

	# convenient version of function for calling from inside Turing
	calc_rv_keplerian(t, p::Vector) = calc_rv_keplerian(t, p...)
end

# ╔═╡ d30799d5-6c82-4987-923e-b8beb2aac74a
begin 
	""" Calculate RV from t, P, K, e, ω, M0	and C   """
	function calc_rv_keplerian_plus_const end 
	
	function calc_rv_keplerian_plus_const(t, P,K,e,ω,M0,C) 
		calc_rv_keplerian(t, P,K,e,ω,M0) + C
	end

	calc_rv_keplerian_plus_const(t, p::Vector) = calc_rv_keplerian_plus_const(t, p...)
end

# ╔═╡ 84a38c60-08a8-425c-b654-8ae5fff8f131
Markdown.parse("""
The eccentric anomaly is related to the mean anomaly by [Kepler's equation](https://en.wikipedia.org/wiki/Kepler%27s_equation).  
A function `calc_ecc_anom` to compute the eccentric anomaly is implemented at the [bottom of the notebook]($(calc_ecc_anom_url)).  
""")

# ╔═╡ b9267ff2-d401-4263-bf25-d52be6260859
md"""
## RV perturbation by planet on a Keplerian orbit
"""

# ╔═╡ 7330040e-1988-4410-b625-74f71f031d43
function simulate_rvs_from_model_v1(chain, times; sample::Integer, chain_id::Integer=1)
	@assert 1<=sample<=size(chain,1)
	@assert 1<=chain_id<=size(chain,3)
	# Extract parameters from chain
	P = chain[sample,:P,chain_id]
	K = chain[sample,:K,chain_id]
	e = chain[sample,:e,chain_id]
	ω = chain[sample,:ω,chain_id]
	M0_minus_ω = chain[sample,:M0_minus_ω,chain_id]
	C = chain[sample,:C,chain_id]

	M0 = M0_minus_ω + ω
	rvs = calc_rv_keplerian_plus_const.(times, P,K,e,ω,M0,C)
end

# ╔═╡ bdac967d-82e0-4d87-82f7-c771896e1797
begin 
	""" Calculate RV from t, P, K, e, ω, M0, C and a	"""
	function calc_rv_keplerian_plus_acc end 
	calc_rv_keplerian_plus_acc(t, p::Vector) = calc_rv_keplerian_plus_acc(t, p...)
	function calc_rv_keplerian_plus_acc(t, P,K,e,ω,M0,C,a) 
		#t0 = bjd_ref::Float64
		calc_rv_keplerian(t, P,K,e,ω,M0) + C + a*t
	end
end

# ╔═╡ 93adb0c3-5c11-479b-9436-8c7df34bd8fe
function simulate_rvs_from_model_v2(chain, times; sample::Integer, chain_id::Integer=1)
	@assert 1<=sample<=size(chain,1)
	@assert 1<=chain_id<=size(chain,3)
	# Extract parameters from chain
	P = chain[sample,:P,chain_id]
	K = chain[sample,:K,chain_id]
	e = chain[sample,:e,chain_id]
	ω = chain[sample,:ω,chain_id]
	M0_minus_ω = chain[sample,:M0_minus_ω,chain_id]
	C = chain[sample,:C,chain_id]
	a = chain[sample,:a,chain_id]
	M0 = M0_minus_ω + ω
	rvs = calc_rv_keplerian_plus_acc.(times, P,K,e,ω,M0,C,a)
end

# ╔═╡ 3fd2ec1a-fed7-43a6-bc0b-ccadb1f711dd
md"""
## A custom prior probability distribution
"""

# ╔═╡ f1547a42-ee3b-44dc-9147-d9c8ec56f1e3
begin
	struct ModifiedJeffreysPriorForScale{T1,T2,T3} <: ContinuousUnivariateDistribution where { T1, T2, T3 }
		scale::T1
		max::T2
		norm::T3
	end
	
	function ModifiedJeffreysPriorForScale(s::T1, m::T2) where { T1, T2 }
		@assert zero(s) < s && !isinf(s)
		@assert zero(m) < m && !isinf(s)
		norm = 1/log1p(m/s)         # Ensure proper normalization
		ModifiedJeffreysPriorForScale{T1,T2,typeof(norm)}(s,m,norm)
	end
	
	function Distributions.rand(rng::AbstractRNG, d::ModifiedJeffreysPriorForScale{T1,T2,T3}) where {T1,T2,T3}
		u = rand(rng)               # sample in [0, 1]
		d.scale*(exp(u/d.norm)-1)   # inverse CDF method for sampling
	end

	function Distributions.logpdf(d::ModifiedJeffreysPriorForScale{T1,T2,T3}, x::Real) where {T1,T2,T3}
		log(d.norm/(1+x/d.scale))
	end
	
	function Distributions.logpdf(d::ModifiedJeffreysPriorForScale{T1,T2,T3}, x::AbstractVector{<:Real})  where {T1,T2,T3}
	    output = zeros(x)
		for (i,z) in enumerate(x)
			output[i] = logpdf(d,z)
		end
		return output
	end
	
	Distributions.minimum(d::ModifiedJeffreysPriorForScale{T1,T2,T3})  where {T1,T2,T3} = zero(T2)
	Distributions.maximum(d::ModifiedJeffreysPriorForScale{T1,T2,T3})  where {T1,T2,T3} = d.max
	
	custom_prob_dist_url = "#" * (PlutoRunner.currently_running_cell_id[] |> string)
end;

# ╔═╡ 62b33a0f-ff96-4eb4-8eaa-0f6c6b7dd42d
Markdown.parse("""
Since this distribution is not provided by the Distributions.jl package, we implement our own [`ModifiedJeffreysPriorForScale`]($custom_prob_dist_url) near the bottom of the notebook.  This may be a useful example for anyone whose class project involves performing statistical inference on data and would like to use a custom distribution.    
""")

# ╔═╡ 83598a97-cf59-4ed9-8c6e-f72a87f4feb6
begin
	P_max = 10*365.25 # 10 years
	K_max = 2129.0     # m/s
	prior_P = ModifiedJeffreysPriorForScale(1.0, P_max)
	prior_K = ModifiedJeffreysPriorForScale(1.0, K_max)
end;

# ╔═╡ 37edd756-e889-491e-8710-a54a862a9cd8
@model rv_kepler_model_v1(t, rv_obs, σ_obs) = begin
	# Specify Priors
	P ~ prior_P                  # orbital period
	K ~ prior_K                  # RV amplitude
	e ~ prior_e                  # orbital eccentricity
	ω ~ Uniform(0, 2π)           # arguement of pericenter
	M0_minus_ω ~ Uniform(0,2π)   # mean anomaly at t=0 minus ω
	C ~ Normal(0,1000.0)         # velocity offset
	σ_j ~ prior_jitter           # magnitude of RV jitter
	
	# Transformations to make sampling more efficient
	M0 = M0_minus_ω + ω

	# Reject any parameter values that are unphysical, _before_ trying 
	# to calculate the likelihood to avoid errors/assertions
	if !(0.0 <= e < 1.0)      
        Turing.@addlogprob! -Inf
        return
    end

	# Likelihood
    # Calculate the true velocity given model parameters
	rv_true = calc_rv_keplerian_plus_const.(t, P,K,e,ω,M0,C)
	
	# Specify measurement model
	σ_eff = sqrt.(σ_obs.^2 .+ σ_j.^2)
 	rv_obs ~ MvNormal(rv_true, σ_eff )
end

# ╔═╡ 776a96af-2c4f-4d6d-9cec-b5db127fed6c
posterior_1 = rv_kepler_model_v1(df.bjd.-bjd_ref,df.rv,df.σ_rv)

# ╔═╡ 4a6008ba-c690-4f62-a914-16c2ce00a103
optimize(posterior_1, MAP(), ConjugateGradient()) 

# ╔═╡ 3a838f95-283b-4e06-b54e-400c1ebe94f8
if  Sys.iswindows() || (Threads.nthreads()==1)
	chains_rand_init = sample(posterior_1, NUTS(), num_steps_per_chain, discard_initial=0)
else
	chains_rand_init = sample(posterior_1, NUTS(), MCMCThreads(), num_steps_per_chain, num_chains, discard_initial=0)
end

# ╔═╡ 6fad4fd9-b99b-44a9-9c45-6e79ffd4a796
traceplot(chains_rand_init,:P)

# ╔═╡ 7a6e2fa8-14db-4031-8560-d735055e7afc
let 
	plt1 = traceplot(chains_rand_init,:K)
	plt2 = traceplot(chains_rand_init,:e)
	plt3 = traceplot(chains_rand_init,:ω)
	plt4 = traceplot(chains_rand_init,:σ_j)
	plot(plt1, plt2, plt3, plt4, layout=(2,2))
end

# ╔═╡ 62c0c2ca-b43b-4c53-a297-08746fae3f6e
describe(chains_rand_init)

# ╔═╡ 1966aff9-ee3d-46f0-be0f-fed723b14f30
md"Chain to calculate summary statistics for: $(@bind chain_id Slider(1:size(chains_rand_init,3);default=1))"

# ╔═╡ 764d36d4-5a3b-48f6-93d8-1e15a44c3ace
md"**Summary statistics for chain $chain_id**"

# ╔═╡ 71962ff1-b769-4601-8b50-7484ca3a0d91
describe(chains_rand_init[:,:,chain_id])

# ╔═╡ c81bb742-a911-4fba-9e85-dfb9a241b290
density(chains_rand_init,:P)

# ╔═╡ 0f672bba-d336-46e7-876d-8b1ce985fa34
map_estimate_with_init_guess = optimize(posterior_1, MAP(), collect(param_guess) )

# ╔═╡ 7ef2f1ca-d542-435b-abdd-03af4b4257f3
if  @isdefined param_guess
	local plt = plot(legend=:bottomright)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv], label=:none)

	rvs_plt = calc_rv_keplerian_plus_const.(t_plt,P_guess,K_guess,e_guess,ω_guess,M0_minus_ω_guess+ω_guess,C_guess)
	plot!(plt,t_plt,rvs_plt, label="Guess for parameters") 

	P_fit, K_fit, e_fit, ω_fit, M0_minus_ω_fit, C_fit = collect(map_estimate_with_init_guess.values)
	rvs_plt = calc_rv_keplerian_plus_const.(t_plt,P_fit,K_fit,e_fit,ω_fit,M0_minus_ω_fit+ω_fit,C_fit)
	plot!(plt,t_plt,rvs_plt,label="MAP starting from guess") 

	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!("Updated RV Predictions") 
end

# ╔═╡ 7a664634-c6e3-464b-90a0-6b4cf5015e83
if go_sample_posterior1 && (P_guess > 0)
	 if Sys.iswindows() || (Threads.nthreads()==1)
		chains_with_guess = sample(posterior_1, NUTS(), num_steps_per_chain*2; init_params = param_guess)
	 else 
		chains_with_guess = sample(posterior_1, NUTS(), MCMCThreads(), num_steps_per_chain*2, num_chains; init_params = fill(param_guess,num_chains))
	 end
end;

# ╔═╡ e9465e3f-f340-4cc5-9fa2-454feaa6bd4d
if @isdefined chains_with_guess
	chain_summary_stats = describe(chains_with_guess)
end

# ╔═╡ 3c3ecf3e-5eba-4bb1-92f7-449680be4edd
if @isdefined chains_with_guess
md"Chain to calculate summary statistics for: $(@bind chain_id2 Slider(1:size(chains_with_guess,3);default=1))"
end

# ╔═╡ b7766545-f1a9-4635-905a-fa3e798f12dc
if go_sample_posterior1 
	md"**Summary statistics for chain $chain_id2**"
end

# ╔═╡ 6441c9cf-f0b1-4229-8b72-1b89e9f0c6f3
if @isdefined chains_with_guess
	describe(chains_with_guess[:,:,chain_id2])
end

# ╔═╡ 3b7580c9-e04e-4768-ac6f-ddb4462dedd8
if go_sample_posterior1 
	density(chains_with_guess)
end

# ╔═╡ 3cfc82d6-3390-4572-b5f0-124503e2e9e0
@model rv_kepler_model_v2(t, rv_obs, σ_obs) = begin
	# Specify Priors
	P ~ prior_P                  # orbital period
	K ~ prior_K                  # RV amplitude
	e ~ prior_e                  # orbital eccentricity
	ω ~ Uniform(0, 2π)           # arguement of pericenter
	M0_minus_ω ~ Uniform(0,2π)   # mean anomaly at t=0 minus ω
	C ~ Normal(0,1000.0)         # velocity offset
	σ_j ~ prior_jitter           # magnitude of RV jitter
	# TODO:  Set prior for a
	
	# Transformations to make sampling easier
	M0 = M0_minus_ω + ω

	# Reject any parameter values that are unphysical, _before_ trying 
	# to calculate the likelihood to avoid errors/assertions
	if !(0.0 <= e < 1.0)      
        Turing.@addlogprob! -Inf
        return
    end
	
    # Calculate the true velocity given model parameters
	# TODO: Update to include an acceleration
	rv_true = calc_rv_keplerian_plus_const.(t, P,K,e,ω,M0,C) 
	
	# Specify model likelihood for the observations
	σ_eff = sqrt.(σ_obs.^2 .+ σ_j.^2)
	rv_obs ~ MvNormal(rv_true, σ_eff )
end

# ╔═╡ fb2fe33c-3854-435e-b9e5-60e8531fd1f3
posterior_2 = rv_kepler_model_v2(df.bjd.-bjd_ref,df.rv,df.σ_rv)

# ╔═╡ 8ffb3e78-515b-492d-bd6f-b24eb08e93d6
if go_sample_posterior2 && (P_guess > 0)
	if Sys.iswindows() || (Threads.nthreads()==1)
		chains_posterior2 = sample(posterior_2, NUTS(), num_steps_per_chain; init_params = param_guess_with_acc)
	else
		chains_posterior2 = sample(posterior_2, NUTS(), MCMCThreads(), num_steps_per_chain, num_chains; init_params = fill(param_guess_with_acc, num_chains))
	end
end;

# ╔═╡ cd1ca1b3-a0ec-4d10-90a3-7648fe52f206
if @isdefined chains_posterior2
	describe(chains_posterior2)
end

# ╔═╡ 9d57a595-fad9-4263-baf4-33d4ac5209f7
if @isdefined chains_posterior2
md"Chain to calculate summary statistics for: $(@bind chain_id3 Slider(1:size(chains_posterior2,3);default=1))"
end

# ╔═╡ 4c05803d-3b8d-4c03-9c7a-3c589227a807
if @isdefined chains_posterior2
	describe(chains_posterior2[:,:,chain_id3])
end

# ╔═╡ 8a4ae52b-9cc6-478f-b836-62e59694949e
if @isdefined chains_posterior2
	density(chains_posterior2)
end

# ╔═╡ 9987e752-164f-40df-98ed-073d715ad87b
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:P)
	local plt2 = density(chains_posterior2,:P)
	title!(plt1,"p(P | data, model w/o acceleration)")
	title!(plt2,"p(P | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ 2fb25751-a036-4156-9fbd-3aaf4e373b91
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:K)
	local plt2 = density(chains_posterior2,:K)
	title!(plt1,"p(K | data, model w/o acceleration)")
	title!(plt2,"p(K | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ 1dfb8d58-b9f3-47a3-a7b1-e8354e7db4e2
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:e)
	local plt2 = density(chains_posterior2,:e)
	title!(plt1,"p(e | data, model w/o acceleration)")
	title!(plt2,"p(e | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ 0f1bf89e-c195-4c5f-9cd9-a2982b2e7bf0
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:ω)
	local plt2 = density(chains_posterior2,:ω)
	title!(plt1,"p(ω | data, model w/o acceleration)")
	title!(plt2,"p(ω | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ df503ec7-a9fa-4170-8892-d19e78c32d39
if  @isdefined chains_rand_init
	draw_new_samples_from_posterior_with_init_from_prior
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_rand_init,1)//2) :
				 size(chains_rand_init,1))
		chain_id =  rand(1:size(chains_rand_init,3))
		rvs = simulate_rvs_from_model_v1(chains_rand_init,t_plt,
					sample=sample_id, 
					chain_id=chain_id)
		plot!(t_plt,rvs) 
	end
	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Predicted RV curves for $n_draws random samples from\nMarkov chains initialized with draws from the prior")
end

# ╔═╡ 6d2e7fd2-3cb8-4e10-ad7c-6d05eb974aa7
if  @isdefined chains_with_guess
	draw_new_samples_from_posterior_with_guess
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_with_guess,1)//2) :
				 size(chains_with_guess,1))
		chain_id =  rand(1:size(chains_with_guess,3))
		rvs = simulate_rvs_from_model_v1(chains_with_guess,t_plt,
					sample=sample_id, 
					chain_id=chain_id)
		plot!(t_plt,rvs) 
	end
	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Predicted RV curves for $n_draws random samples from\nnew Markov chains")

end

# ╔═╡ 693cae36-613b-4c3d-b6a0-3284b1831520
if  @isdefined chains_posterior2
	draw_new_samples_from_posterior_with_acc
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_posterior2,1)//2) :
				 size(chains_posterior2,1))
		chain_id =  rand(1:size(chains_posterior2,3))
		rvs = simulate_rvs_from_model_v2(chains_posterior2,t_plt,
					sample=sample_id, 
					chain_id=chain_id)
		plot!(t_plt,rvs) 
	end
	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Predicted RV curves for $n_draws random samples from\nMarkov chains for model with acceleration term")

end

# ╔═╡ fe8f637d-3721-4a9f-9e6e-f6aee00b7f18
if  @isdefined chains_posterior2
	draw_new_samples_from_posterior_with_guess
	local plt = standardize_histo ? plot(Normal(0,1),legend=:none, color=:black, lw=3) : plot() 
	local resid = zeros(length(df.bjd),n_draws)
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_with_guess,1)//2) :
				 size(chains_with_guess,1))
		chain_id =  rand(1:size(chains_with_guess,3))
		rvs_pred = simulate_rvs_from_model_v1(chains_with_guess,df.bjd.-bjd_ref,
					sample=sample_id, 
					chain_id=chain_id)
		resid[:,i] .= (df.rv.-rvs_pred)
		if standardize_histo
			resid[:,i] ./= sqrt.(df.σ_rv.^2 .+ chains_with_guess[sample_id,:σ_j,chain_id]^2)
		end
	end
	
	histogram!(vec(resid), bins=32, alpha=0.5, label="w/o acc", color=1, normalize=true)
	
	#resid = zeros(length(df.bjd),n_draws)
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_posterior2,1)//2) :
				 size(chains_posterior2,1))
		chain_id =  rand(1:size(chains_posterior2,3))
		rvs_pred = simulate_rvs_from_model_v2(chains_posterior2,df.bjd.-bjd_ref,
					sample=sample_id, 
					chain_id=chain_id)
		resid[:,i] .= (df.rv.-rvs_pred)
		if standardize_histo
			resid[:,i] ./= sqrt.(df.σ_rv.^2 .+ chains_posterior2[sample_id,:σ_j,chain_id]^2)
		end
	end
	
	histogram!(vec(resid), bins=32, alpha=0.5, normalize=true, color=2, label="w/ acc")
	if standardize_histo
		title!(plt,"Histogram of standarized residuals")
		xlabel!(plt,"Standardized Residuals")
	else
		title!(plt,"Histogram of residuals")
		xlabel!(plt,"Residuals (m/s)")
		end
	ylabel!(plt,"Density")
end

# ╔═╡ 0bb0c056-cb48-4ed7-8305-24d2957b595a
md"""
## Compute a Periodorgam
"""

# ╔═╡ bfc03760-17bb-4dac-a233-47dd3827519c
md"### Compute design matrix for periodograms"

# ╔═╡ 8b662b76-79d3-41ff-b5e0-fd06163ad5f8
function calc_design_matrix_circ!(result::AM, period, times::AV) where { R1<:Real, AM<:AbstractMatrix{R1}, AV<:AbstractVector{R1} }
	n = length(times)
	@assert size(result) == (n, 2)
	for i in 1:n
		( result[i,1], result[i,2] ) = sincos(2π/period .* times[i])
	end
	return result
end

# ╔═╡ 61c7e285-6dd4-4811-8016-45c863fdb397
function calc_design_matrix_circ(period, times::AV) where { R1<:Real, AV<:AbstractVector{R1} }
	n = length(times)
	dm = zeros(n,2)
	calc_design_matrix_circ!(dm,period,times)
	return dm
end

# ╔═╡ e887b6ee-9f57-4629-ab31-c74d80cb948a
function calc_design_matrix_lowe!(result::AM, period, times::AV) where { R1<:Real, AM<:AbstractMatrix{R1}, AV<:AbstractVector{R1} }
	n = length(times)
	@assert size(result) == (n, 4)
	for i in 1:n
		arg = 2π/period .* times[i]
		( result[i,1], result[i,2] ) = sincos(arg)
		( result[i,3], result[i,4] ) = sincos(2*arg)
	end
	return result
end

# ╔═╡ 327391cf-864a-4c82-8aa9-d435fe44d0e1
function calc_design_matrix_lowe(period, times::AV) where { R1<:Real, AV<:AbstractVector{R1} }
	n = length(times)
	dm = zeros(n,4)
	calc_design_matrix_lowe!(dm,period,times)
	return dm
end

# ╔═╡ c9687fcb-a1e1-442c-a3c7-f0f60350b059
md"## Generalized Linear Least Squares Fitting" 

# ╔═╡ 6f820e58-b61e-43c0-95dc-6d0e936f71c3
function fit_general_linear_least_squares( design_mat::ADM, covar_mat::APD, obs::AA ) where { ADM<:AbstractMatrix, APD<:AbstractPDMat, AA<:AbstractArray }
	Xt_inv_covar_X = Xt_invA_X(covar_mat,design_mat)
	X_inv_covar_y =  design_mat' * (covar_mat \ obs)
	AB_hat = Xt_inv_covar_X \ X_inv_covar_y                            # standard GLS
end

# ╔═╡ b1bfd41e-3335-46ef-be5a-2aab2532060f
function predict_general_linear_least_squares( design_mat::ADM, covar_mat::APD, obs::AA ) where { ADM<:AbstractMatrix, APD<:AbstractPDMat, AA<:AbstractArray }
	param = fit_general_linear_least_squares(design_mat,covar_mat,obs)
	design_mat * param 
end

# ╔═╡ 1d3d4e92-e21d-43f8-b7f4-5191d8d42821
function calc_χ²_general_linear_least_squares( design_mat::ADM, covar_mat::APD, obs::AA ) where { ADM<:AbstractMatrix, APD<:AbstractPDMat, AA<:AbstractArray }
	pred = predict_general_linear_least_squares(design_mat,covar_mat,obs)
	invquad(covar_mat,obs-pred)
end

# ╔═╡ 42fd9719-26a3-4742-974a-303eb5e810c5
function calc_periodogram(t, y_obs, covar_mat; period_min::Real = 2.0, period_max::Real = 4*(maximum(t)-minimum(t)), num_periods::Integer = 4000)
	period_grid =  1.0 ./ range(1.0/period_max, stop=1.0/period_min, length=num_periods) 
	periodogram = map(p->-0.5*calc_χ²_general_linear_least_squares(calc_design_matrix_circ(p,t),covar_mat,y_obs),period_grid)
	period_fit = period_grid[argmax(periodogram)]
	design_matrix_fit = calc_design_matrix_circ(period_fit,t)
	coeff_fit = fit_general_linear_least_squares(design_matrix_fit,covar_mat,y_obs)
	phase_fit = atan(coeff_fit[1],coeff_fit[2])
	pred = design_matrix_fit * coeff_fit
	rms = sqrt(mean((y_obs.-pred).^2))
	return (;period_grid=period_grid, periodogram=periodogram, period_best_fit = period_fit, coeff_best_fit=coeff_fit, phase_best_fit=phase_fit, predict=pred, rms=rms )
end
	

# ╔═╡ 2ccca815-bd26-4f0f-b966-b2ab2fe02d01
begin
	jitter_for_periodogram = 3.0
	num_period_for_periodogram = 10_000
	periodogram_results = calc_periodogram(df.bjd.-bjd_ref,df.rv,
								PDiagMat(sqrt.(df.σ_rv.^2 .+jitter_for_periodogram^2)), 
									num_periods=num_period_for_periodogram)
end;

# ╔═╡ a5be518a-3c7c-424e-b100-ec8967f4ae27
plot(periodogram_results.period_grid,periodogram_results.periodogram, xscale=:log10, xlabel="Putative Orbital Period (d)", ylabel="-χ²/2", legend=:none)

# ╔═╡ cc51e4bd-896f-479c-8d09-0ce3f07e402c
md"## Get log probability from Turing model"

# ╔═╡ f3fb26a5-46b5-4ba3-8a30-9a88d6868a24

function make_logp(
    model::Turing.Model,
    sampler=Turing.SampleFromPrior(),
    ctx::Turing.DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    vi = Turing.VarInfo(model)

    # define function to compute log joint.
    function ℓ(θ)
        new_vi = Turing.VarInfo(vi, sampler, θ)
        model(new_vi, sampler, ctx)
        logp = Turing.getlogp(new_vi)
        return logp
    end

end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.5"
Distributions = "~0.25.71"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
MCMCChains = "~5.3.1"
Optim = "~1.7.3"
PDMats = "~0.11.16"
Plots = "~1.33.0"
PlutoTeachingTools = "~0.2.3"
PlutoUI = "~0.7.40"
StatsPlots = "~0.15.3"
Turing = "~0.21.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "5c26c7759412ffcaf0dd6e3172e55d783dd7610b"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.1.3"

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
git-tree-sha1 = "5c0b629df8a5566a06f5fef5100b53ea56e465a0"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "0091e2e4d0a7125da0e3ad8c7dbff9171a921461"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.6"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "d7a7dabeaef34e5106cdf6c2ac956e9e3f97f666"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.8"

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

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5bb0f8292405a516880a3809954cb832ae7a31c5"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.20"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

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
git-tree-sha1 = "a3704b8e5170f9339dff4e6cb286ad49464d3646"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

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
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "a5fd229d3569a6600ae47abe8cd48cbeb972e173"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.44.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dc4405cee4b2fe9e1108caec2d760b7ea758eca2"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.5"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

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
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

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
git-tree-sha1 = "332a332c97c7071600984b3c31d9067e1a4e6e25"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "78bee250c6826e1cf805a88b7f1e86025275d208"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.0"

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
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

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
git-tree-sha1 = "6bce52b2060598d8caaed807ec6d6da2a1de949e"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.5"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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
git-tree-sha1 = "992a23afdb109d0d2f8802a30cf5ae4b1fe7ea68"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.1"

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
git-tree-sha1 = "ee407ce31ab2f1bacadc3bd987e96de17e00aed3"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.71"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "74dd5dac82812af7041ae322584d5c2181dead5c"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.42"

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
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "MacroTools", "OrderedCollections", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "7bc3920ba1e577ad3d7ebac75602ab42b557e28e"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.20.2"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterfaceCore", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4cda4527e990c0cc201286e0a0bfbbce00abcfc2"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.0.0"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "87519eb762f85534445f5cda35be12e32759ee14"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.4"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "5a2cff9b6b77b33b89f3d97a4d367747adce647e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.15.0"

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
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

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

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "a5e6e7f12607e90d71b09e6ce2c965e41b337968"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.1"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "a2657dd0f3e8a61dbe70fc7c122038bd33790af5"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.3.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3697c23d09d5ec6f2088faa68f0d926b6889b5be"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.67.0+0"

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
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "59ba44e0aa49b87a8c7a8920ec76f8afe87ed502"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.3.3"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

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
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

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
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "f67b55b6447d36733596aea445a9f119e83498b6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.5"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "076bb0da51a8c8d1229936a1af7bdfacd65037e1"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.2"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

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
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

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
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "dfa6c5f2d5a8918dd97c7f1a9ea0de68c2365426"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.7.5"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random", "Requires", "UnPack"]
git-tree-sha1 = "408a29d70f8032b50b22155e6d7776715144b761"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "1.0.2"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8cb9b8fb081afd7728f5de25b9025bff97cb5c7a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.3.1"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "59ac3cc5c08023f58b9cd6a5c447c4407cede6bc"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "41d162ae9c868218b1f3fe78cba878aa348c2d26"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.1.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "16fa7c2e14aa5b3854bc77ab5f1dbe2cdc488903"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.6.0"

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
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "6872f9594ff273da6d13c7c1a1545d5a8c7d0c1c"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.6"

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
git-tree-sha1 = "efe9c8ecab7a6311d4b91568bd6c88897822fabe"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "415108fd88d6f55cedf7ee940c7d4b01fad85421"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.9"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

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
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "b9fe76d1a39807fdcf790b991981a922de0c3050"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.3"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1ef34738708e3f31994b52693286dabcb3d29f6b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.9"

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
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

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
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6062b3b25ad3c58e817df0747fc51518b9110e5f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.33.0"

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
git-tree-sha1 = "a602d7b0babfca89005da04d89223b867b55319f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.40"

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
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

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
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "3004608dc42101a944e44c1c68b599fa7c669080"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

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
git-tree-sha1 = "50f945fb7d7fdece03bbc76ff1ab96170f64a892"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "b04da5c714e0eb117c508055dc2f3d9b4f46a45e"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.57.1"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

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

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

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
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "efa8acd030667776248eabb054b1836ac81d92f0"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.7"

[[deps.StaticArraysCore]]
git-tree-sha1 = "ec2bd695e905a3c755b33026954b119ea17f2d22"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.3.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3e59e005c5caeb1a57a90b17f582cbfc2c8da8f7"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.3"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

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
git-tree-sha1 = "4d5536136ca85fe9931d6e8920c138bb9fcc6532"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.8.0"

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
git-tree-sha1 = "f53e34e784ae771eb9ccde4d72e578aa453d0554"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.6"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d963aad627fd7af56fbbfee67703c2f7bfee9dd7"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.22"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "68fb67dab0c11de2bb1d761d7a742b965a9bc875"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.12"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

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
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

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
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

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

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

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
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─10709dbc-c38b-4f15-8ea8-772db2acfbb3
# ╟─47a0c0de-7d75-4891-b671-c083578a004d
# ╟─73021015-4357-4196-868c-e9564de02ede
# ╟─13d4e147-d12a-43e5-883d-b037a8e3b433
# ╟─0574750f-04d1-4677-b52d-ac3aa68c6eee
# ╠═0ad398fb-9c7e-467d-a932-75db70cd2e86
# ╟─f6512ab6-2783-43c1-81a5-9c05c539b749
# ╠═1492e4af-440d-4926-a4b5-b33da77dbee2
# ╟─84a38c60-08a8-425c-b654-8ae5fff8f131
# ╟─9f7196d2-83c4-4e6a-baa7-b8a17f51b6e3
# ╠═d30799d5-6c82-4987-923e-b8beb2aac74a
# ╟─a85268d5-40a6-4654-a4be-cba380e97d35
# ╟─cfe8d6ad-2125-4587-af70-875e7c4c4844
# ╟─b9b14b8d-f843-48f8-afe5-4d8f3008460a
# ╟─9ffab6aa-3e9c-4daf-91f5-af3971528341
# ╟─9a7302cd-3b51-4b1e-b932-6b5785998a8a
# ╟─54467bb4-6901-4934-8c11-7f32012406c8
# ╠═b289d8f7-b138-43e7-98a1-d971089d7f72
# ╟─21612349-dbc5-4bdd-8036-12900a005b21
# ╟─62b33a0f-ff96-4eb4-8eaa-0f6c6b7dd42d
# ╠═83598a97-cf59-4ed9-8c6e-f72a87f4feb6
# ╟─a1a7d4b7-8c02-4e74-a158-3e05b4da63fb
# ╠═42589e2f-3b4d-4764-8f96-00614ed144e7
# ╟─a2140dbe-8736-4ed9-ae6f-b1b0c7df3bc9
# ╟─22719976-86f5-43d3-b890-d3520f9916d2
# ╟─e50bdd14-d855-4043-bbab-f6526a972e31
# ╟─87dd11ad-c29e-4a5e-90b1-289863eedd57
# ╟─ddac2663-fd90-4c60-acaa-da2560367706
# ╟─b9efe114-717e-46c8-8225-a4ab4d3df439
# ╠═2844cd3a-9ed1-47da-ab59-ea33575b4991
# ╟─f28d4bc8-53e7-45f9-8126-7339a6f54732
# ╟─342a0868-4f28-444d-b858-91ccdc655561
# ╟─56aa2fd5-fb34-495c-9b9f-0ce0bbbd6b1b
# ╠═37edd756-e889-491e-8710-a54a862a9cd8
# ╟─228bb255-319c-4e80-95b3-8bf333be29e4
# ╠═776a96af-2c4f-4d6d-9cec-b5db127fed6c
# ╟─53240de7-f147-485c-b113-ff20aaa1227b
# ╠═4a6008ba-c690-4f62-a914-16c2ce00a103
# ╟─60b2a73a-e0dc-4e9c-a078-b3fd34f0def7
# ╠═795011db-ec97-4f2b-907d-14fa35094b54
# ╟─1e143b7b-3598-4048-9133-2110944b3a67
# ╟─5ab4a787-fe9b-4b2c-b14c-273e0205259d
# ╟─5f20e01d-3489-4828-9e4e-119095e9c29c
# ╟─dfdf843d-98ce-40a1-bd0b-0a11f1cdb5f9
# ╟─87a2281b-8827-4eca-89a6-c7d77aa7a00f
# ╟─9f27ff9f-c2c4-4bf4-b4cd-bac713de0abe
# ╠═3a838f95-283b-4e06-b54e-400c1ebe94f8
# ╟─358ab0f4-3423-4728-adfa-44126744ae18
# ╟─c1fa4faa-f10c-4ed0-9539-20f5fcde8763
# ╟─00c4e783-841c-4d6a-a53d-34a88bbe1f7a
# ╠═6fad4fd9-b99b-44a9-9c45-6e79ffd4a796
# ╟─7a6e2fa8-14db-4031-8560-d735055e7afc
# ╟─f7046ee4-9eb7-4324-886e-5a3650b58db7
# ╟─bb2ff977-2d12-4baa-802c-097a4138a24b
# ╠═62c0c2ca-b43b-4c53-a297-08746fae3f6e
# ╟─1196e7e6-4144-47e8-97cd-6f93530e11e5
# ╟─55a3378a-da18-4310-8272-cf11298a118d
# ╟─1966aff9-ee3d-46f0-be0f-fed723b14f30
# ╟─764d36d4-5a3b-48f6-93d8-1e15a44c3ace
# ╠═71962ff1-b769-4601-8b50-7484ca3a0d91
# ╟─ac237181-b6d2-4968-b6e3-146ba41bcc18
# ╠═c81bb742-a911-4fba-9e85-dfb9a241b290
# ╟─a8e547c5-c2b5-4058-a1e2-35397d7e8112
# ╠═80d076a8-7fb1-4dc7-8e3b-7579db30b038
# ╟─3759ec08-14f4-4026-916b-6a0bbc818ca7
# ╟─2fa23729-007e-4dcb-b6f9-67407929c53a
# ╟─52f4f8bd-41a9-446f-87e7-cdc1e9c178f8
# ╟─bf5b40c5-399e-40a0-8c27-335f72015e58
# ╟─98bc7209-a6a1-4c06-8e6f-de5b421b7b8c
# ╟─207a0f69-30ce-4107-a21d-ba9f6c5a5039
# ╟─df503ec7-a9fa-4170-8892-d19e78c32d39
# ╟─93115337-ec60-43b9-a0ef-1d3ffc1fe5ff
# ╟─0d83ebc9-ecb3-4a9d-869b-351b28d5e962
# ╠═cc2dbf2f-27a4-4cd6-81e5-4a9f5a3bc488
# ╟─f00539c9-ee40-4128-928c-460fd439fa87
# ╟─b787d7ef-bda6-470d-93b8-954227dec24b
# ╟─aaf24ba9-48dd-49ff-b760-2b806f7cf5ba
# ╟─2ccca815-bd26-4f0f-b966-b2ab2fe02d01
# ╟─a5be518a-3c7c-424e-b100-ec8967f4ae27
# ╟─e62067ce-dbf6-4c50-94e3-fce0b8cbef12
# ╟─dd918d00-fbe7-4bba-ad6a-e77ea0acb43a
# ╟─9c82e944-d5f2-4d81-9b33-7e618421b715
# ╠═b76b3b7f-c5f3-4aec-99b5-799479c124d6
# ╟─0085a6ce-334a-46a7-a12d-ee7b61c59f23
# ╠═0f672bba-d336-46e7-876d-8b1ce985fa34
# ╟─3459436a-bfeb-45d4-950e-68fd55af76d7
# ╟─7ef2f1ca-d542-435b-abdd-03af4b4257f3
# ╟─47e61e3a-466d-40f9-892e-5691eb6a2393
# ╟─ba4e0dc6-5008-4ed8-8eba-3902820cf241
# ╟─a247788c-8072-4cc4-8c38-68d0a3445e83
# ╠═7a664634-c6e3-464b-90a0-6b4cf5015e83
# ╟─565d68a4-24cd-479a-82ab-21d64a6a01f6
# ╟─db2ba5d1-3bd3-4208-9805-9c3fab259377
# ╟─e9465e3f-f340-4cc5-9fa2-454feaa6bd4d
# ╟─3c3ecf3e-5eba-4bb1-92f7-449680be4edd
# ╟─b7766545-f1a9-4635-905a-fa3e798f12dc
# ╟─6441c9cf-f0b1-4229-8b72-1b89e9f0c6f3
# ╟─c9433de9-4f48-4bc6-846c-3d684ae6adee
# ╠═3b7580c9-e04e-4768-ac6f-ddb4462dedd8
# ╟─7f8394ee-e6a2-4e4f-84b2-10a043b3da35
# ╠═f2ff4d87-db0a-4418-a548-a9f3f04f93cd
# ╟─42fe9282-6a37-45f5-a833-d2f6eb0518fe
# ╟─b38f3767-cd14-497c-9576-22764c53a48d
# ╟─35562045-abba-4f36-be92-c41f71591b1a
# ╟─e9530014-1186-4897-b875-d9980e0c3ace
# ╟─6d2e7fd2-3cb8-4e10-ad7c-6d05eb974aa7
# ╟─9dc13414-8266-4f4d-94dd-f11c0726c539
# ╟─95844ef7-8730-469f-b69b-d7bbe5fc2607
# ╠═3699772e-27d4-402e-a669-00f5b22f2ed5
# ╟─9eccb683-d637-4ab9-8af7-70c24a7d8478
# ╟─9a2952be-89d8-41ef-85ce-310ed90bd0d1
# ╟─c7e9649d-496d-44c2-874c-5f51e311b21d
# ╠═3cfc82d6-3390-4572-b5f0-124503e2e9e0
# ╟─5084b413-1211-490a-89d3-1cc782f1741e
# ╟─c6939875-0efc-4f1d-a2d3-6b46484328a5
# ╠═fb2fe33c-3854-435e-b9e5-60e8531fd1f3
# ╟─43177135-f859-423d-b70c-e755fdd06765
# ╠═c33d9e9d-bf00-4fa4-9f90-33415385507e
# ╟─20393a82-6fb1-4583-b7b7-8d1cda43bd47
# ╠═8ffb3e78-515b-492d-bd6f-b24eb08e93d6
# ╟─13c56edd-3c1e-496e-8116-fb158dd0f133
# ╟─c2b7dfcc-6238-4e9a-a2b8-efcc05c17361
# ╟─cd1ca1b3-a0ec-4d10-90a3-7648fe52f206
# ╟─9d57a595-fad9-4263-baf4-33d4ac5209f7
# ╟─4c05803d-3b8d-4c03-9c7a-3c589227a807
# ╟─8a4ae52b-9cc6-478f-b836-62e59694949e
# ╟─21fdeaff-0c91-481c-bd8e-1dba27e275a6
# ╠═a4be55ab-3e8e-423c-b3bc-3f3b88c5d2b7
# ╟─5efb1eac-c1fe-417f-828a-3cfb8978da40
# ╟─bcedc85f-8bf2-49d4-a60a-d6020450fd76
# ╟─d751f93c-9ef4-41ad-967b-7ccde6e40afd
# ╟─693cae36-613b-4c3d-b6a0-3284b1831520
# ╟─5171f2f0-e60c-4038-829f-9baf2d5f178e
# ╟─d34e6593-d847-4728-addb-4d2ebe32fdc0
# ╠═47be22e6-4d81-4849-a4f8-08f4d4537829
# ╟─68134fbb-5e31-4abf-babe-1508d7c692df
# ╟─fe8f637d-3721-4a9f-9e6e-f6aee00b7f18
# ╟─083e9418-9b64-46d0-8da4-3396fb958862
# ╟─8180b43d-81aa-4be0-bdf1-ac93f734331c
# ╟─d9cd7102-490d-4f35-a254-816c069d3810
# ╟─9987e752-164f-40df-98ed-073d715ad87b
# ╟─2fb25751-a036-4156-9fbd-3aaf4e373b91
# ╟─1dfb8d58-b9f3-47a3-a7b1-e8354e7db4e2
# ╟─0f1bf89e-c195-4c5f-9cd9-a2982b2e7bf0
# ╠═9fd50ada-702f-4ca4-aab2-abfa0f4f597c
# ╟─b3f9c7b7-5ed5-47d7-811c-6f4a313de24b
# ╟─461416e4-690f-4ebd-9f07-3e34962c8693
# ╟─e5d0b1cc-ea7b-42a4-bfcd-684337b0f98b
# ╠═5e5d4560-fa1e-48f6-abe4-3b1221d44609
# ╟─dd40a3cf-76d3-4eb9-8027-274a065c762c
# ╟─940f4f42-7bc3-48d4-b9f4-22f9b94c345d
# ╟─40923752-9215-45c9-a444-5a519b64df97
# ╠═6fd32120-4df1-4f2d-bb6f-c348a6999ad5
# ╟─83485386-90fd-4b2d-bdad-070835d8fb44
# ╠═b01b2df0-9cfd-45c8-ba35-cd9ec018af6a
# ╠═f104183b-ea56-45c3-987e-94e42d687143
# ╠═097fab0c-edfc-4d3a-abb2-4285b026e3f2
# ╠═9c50e9eb-39a0-441a-b03f-6358caa2d0e9
# ╟─b9267ff2-d401-4263-bf25-d52be6260859
# ╠═7330040e-1988-4410-b625-74f71f031d43
# ╠═bdac967d-82e0-4d87-82f7-c771896e1797
# ╠═93adb0c3-5c11-479b-9436-8c7df34bd8fe
# ╟─3fd2ec1a-fed7-43a6-bc0b-ccadb1f711dd
# ╠═f1547a42-ee3b-44dc-9147-d9c8ec56f1e3
# ╟─0bb0c056-cb48-4ed7-8305-24d2957b595a
# ╠═42fd9719-26a3-4742-974a-303eb5e810c5
# ╟─bfc03760-17bb-4dac-a233-47dd3827519c
# ╠═8b662b76-79d3-41ff-b5e0-fd06163ad5f8
# ╠═61c7e285-6dd4-4811-8016-45c863fdb397
# ╠═e887b6ee-9f57-4629-ab31-c74d80cb948a
# ╠═327391cf-864a-4c82-8aa9-d435fe44d0e1
# ╟─c9687fcb-a1e1-442c-a3c7-f0f60350b059
# ╠═6f820e58-b61e-43c0-95dc-6d0e936f71c3
# ╠═b1bfd41e-3335-46ef-be5a-2aab2532060f
# ╠═1d3d4e92-e21d-43f8-b7f4-5191d8d42821
# ╟─cc51e4bd-896f-479c-8d09-0ce3f07e402c
# ╠═f3fb26a5-46b5-4ba3-8a30-9a88d6868a24
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
