import os, sys, json, hashlib
import numpy as np

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import torch

sys.path.append(os.path.join(sys.path[0], "backend"))

import backend.sampler
import backend.sd

supportedModel = [".ckpt"]

try:
	import safetensors.torch
	supportedModel += [".safetensors"]
except:
	print("[!] Safetensors not supported!")

def filterExtension(files, extensions):
	return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions, files)))

class EncodeCLIP:
	@classmethod

	def inputType(x):
		return {"required": {
				"text": ("STRING", {"multiline": True}),
				"clip": ("CLIP", )
		}}

		returnType = ("CONDITIONING", )
		function = "encode"

		def encode(self, clip, text):
			return (clip.encode(text), )

class DecodeVAE:
	def __init__(self, device="cpu"):
		self.device = device

	@classmethod

	def inputType(x):
		return {"required": {
				"samples": ("LATENT", ),
				"vae": ("VAE", )
		}}

		returnType = ("IMAGE", )
		function = "decode"

		def decode(self, vae, sample):
			return(vae.decode(sample), )

class EncodeVAE:
	def __init__(self, device="cpu"):
		self.device = device

	@classmethod

	def inputType(x):
		return {"required": {
			"image": ("IMAGE", ),
			"vae": ("VAE", ),
		}}

		returnType = ("LATENT", )
		function = "encode"

	def encode(self, vae, image):
		x = (image.shape[1] // 64) * 64
		y = (image.shape[2] // 64) * 64

		if image.shape[1] != x or image.shape[2] != y:
			image = image[:,:x,:y,:]

		return(vae.encode(image), )

class LoadModel:
	def __init__(self):
		self.configs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
		self.checkpoints = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

	@classmethod
	def inputType(x):
		return {"required": {
				"config": (filterExtension(os.listdir(x.configs), ".yaml"), ),
				"model": (filterExtension(os.listdir(x.models), supportedModel), ),
		}}

		returnType = ("MODEL", "CLIP", "VAE")
		function = "load"

		def load(self, config, model, outputVAE=True, outputCLIP=True):
			configPath = os.path.join(self.configs, config)
			modelPath = os.path.join(self.models, model)
			return backend.sd.load_checkpoint(configPath, modelPath, output_vae=outputVAE, output_clip=outputCLIP)

class LoadVAE:
	def __init__(self):
		self.folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vae")

	@classmethod
	def inputType(x):
		return {"required":
				{"name": (filterExtension(os.listdir(x.folder), supportedModel), )}
				}

	returnType = ("VAE", )
	function = "load"

	def load(self, name):
		path = os.path.join(self.folder, name)
		vae = backend.sd.VAE(path=path)
		return(vae,)

class EmptyLatent:
	def __init__(self, device="cpu"):
		self.device = device

	@classmethod
	def inputType(x):
		return {"required": 
				{"width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
				"height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
				"batchSize": ("INT", {"default": 1, "min": 1, "max": 64})}
		}

	returnType = ("LATENT", )
	function = "generate"

	def generate(self, width, height, batchSize=1):
		latent = torch.zeros([batchSize, 4, height // 8, width // 8])
		return (latent, )

class UpscaleLatent:
	methods = ["nearest-exact", "bilinear", "area"]

	@classmethod
	def inputType(x):
		return {"required":{
				"samples": ("LATENT", ), 
				"method": (x.methods, ),
				"width": ("INT", {"default": 512, "min":64, "max": 4096, "step": 64}),
				"height": ("INT", {"default": 512, "min":64, "max": 4096, "step": 64}),
		}}

	returnType = ("LATENT", )
	function = "upscale"

	def upscale(self, samples, method, width, height):
		upscale = torch.nn.functional.interpolate(samples, size=(height // 8, width // 8), mode=method)
		return (upscale, )

class KSampler:
	def __init__(self, device="cuda"):
		self.device = device

	@classmethod
	def inputType(x):
		return {"required":
				{"model": ("MODEL",),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
				"name": (backend.sampler.KSampler.samplers, ),
				"scheduler": (backend.sampler.KSampler.schedulers, ),
				"positive": ("CONDITIONING", ),
				"negative": ("CONDITIONING", ),
				"latent": ("LATENT", ),
				"denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),


		}}
	returnType = ("LATENT", )
	function = "sample"

	def sample(self, model, seed, steps, cfg, name, scheduler, positive, negative, latent, denoise=1.0):
		noise = torch.randn(latent.size(), dtype=latent.dtype, layout=latent.layout, generator=torch.manual_seed(seed), device="cpu")
		model = model.to(self.device)
		noise = noise.to(self.device)
		latent = latent.to(self.device)

		if positive.shape[0] < noise.shape[0]:
			positive = torch.cat([positive] * noise.shape[0])

		if negative.shape[0] < noise.shape[0]:
			negative = torch.cat([negative] * noise.shape[0])

		positive = positive.to(self.device)
		negative = negative.to(self.device)

		if name in backend.sampler.KSampler.samplers:
			sampler = backend.sampler.KSampler(model, steps=steps, device=self.device, sampler=name, scheduler=scheduler, denoise=denoise)
		else:
			#other samplers
			pass

		sample = sampler.sample(noise, positive, negative, cfg=cfg, latent=latent)
		sample = sample.cpu()
		model = model.cpu()
		return (sample, )

class ImageSave:
	def __init__(self):
		self.folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
		self.prefix = "StellarUI"
		
		try:
			self.counter = int(max(filter(lambda a: f"{self.prefix}_" in a, os.listdir(self.folder))).split("_")[-1]) + 1
		except:
			self.counter = 1

	@classmethod
	def inputType(x):
		return {"required":
			{"images": ("IMAGE", )},
			"hidden": {"prompt": "PROMPT", "info": "INFO"}
		}

	returnType = ()
	function = "save"

	outputNode = True

	def save(self, images, prompt=None, info=None):
		for image in images:
			i = 255. * image.cpu().numpy()
			img = Image.fromarray(i.astype(np.uint8))
			metadata = PngInfo()

			if prompt is not None:
				metadata.add_text("prompt", json.dumps(prompt))
			if info is not None:
				for x in info:
					metadata.add_text(x, json.dumps(info[x]))
			img.save(f"{self.folder}/{self.prefix}_{self.counter:05}.png", pnginfo=metadata, optimize=True)
			self.counter += 1

class LoadImage:
	folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")

	@classmethod
	def inputType(x):
		return {"required": {
				"image": (os.listdir(x.folder),)
		}}

	returnType = ("IMAGE", )
	function = "load"

	def load(self, image):
		path = os.path.join(self.folder, image)
		image = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
		return torch.from_numpy(image[None])[None,]

	@classmethod
	def changed(x, image):
		path = os.path.join(x.folder, image)
		hasher = hashlib.sha256()

		with open(path, "rb") as f:
			hasher.update(f.read())

		return hasher.digest().hex()



moduleMap = {
	"KSampler": KSampler,
	"LoadModel": LoadModel,
	"EncodeCLIP": EncodeCLIP,
	"DecodeVAE": DecodeVAE,
	"EncodeVAE": EncodeVAE,
	"LoadVAE": LoadVAE,
	"EmptyLatent": EmptyLatent,
	"UpscaleLatent": UpscaleLatent,
	"ImageSave": ImageSave,
	"LoadImage": LoadImage
}
