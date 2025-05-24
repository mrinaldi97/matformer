"""
This file includes all the necessary logic for Latent Diffusion models


"""

"""
torch.linspace creates n evenly spaced values between the starting and ending beta points.
N is the number of max. timesteps, while the values are squared so that the noise starts small and then increase
more rapidly as timesteps increase
"""
def initialization(): #Of course it will not be a function, it is just a placeholder for code
	beta_start=0.0
	beta_end=0.0
	n_timesteps=1000
	schedule_betas=torch.linspace(beta_start**0.5,beta_end**0.5,n_timesteps) 
	# Compute the alphas (1-betas)
	alphas=1-betas
	# Cumulative product of alphas, the one that is going to be used in the formula
	alphas_cumprod=torch.cumprod(alphas,dim=0).to(device)
	pass

def noiser(): #Of course it will not be a function, it is just a placeholder for code
	random_timestep=torch.randint(0, num_train_timesteps, (latents.shape[0],), device=device).long() #Picking a random timestep for noise
	noise_epsilon = torch.randn_like(latent) #Generating random gaussian noise
	sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]) #Return shape [batch_size]
	sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t]) #Return shape [batch_size]

	# Reshape to allow broadcasting with latents: (batch_size, 1, 1, 1)
	sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
	sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)

	# x_t = sqrt(bar_alpha_t)*x_0 + sqrt(1-bar_alpha_t)*epsilon
	noisy_latents = sqrt_alphas_cumprod_t * latents + sqrt_one_minus_alphas_cumprod_t * noise_epsilon	
	
