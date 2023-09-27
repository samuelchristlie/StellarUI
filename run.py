import os, sys, copy, json, threading, queue, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import modules


def inputData(inputs, objectClass, output={}, prompt={}, data={}):
	inputType = objectClass.inputType()
	allData = {}

	for i in inputs:
		data = inputs[i]

		if isinstance(data, list):
			inputIdentifier = data[0]
			outputIndex = data[1]

			outputObject = output[inputIdentifier][outputIndex]
			allData[i] = outputObject
		else:
			if ("required" in inputType and i in inputType["required"]) or ("optional" in inputType and i in inputType["optional"]):
				allData = data

	

	if "hidden" in inputType:
		hidden = inputType["hidden"]

		for i in hidden:
			if hidden[i] == "PROMPT":
				allData[i] = prompt
			if hidden[i] == "info":
				if "info" in data:
					allData[i] = data["info"]
	return allData

def recursiveExecute(prompt, output, current, data={}):
	inputs = prompt[current]["inputs"]
	classType = prompt[current]["classType"]

	objectClass = modules.moduleMap[classType]
	inputType = objectClass.inputType()

	if current in output:
		return []

	executed = []

	for i in inputs:
		data = inputs[i]

		if isinstance(data, list):
			inputIdentifier = data[0]
			outputIndex = data[1]

			if inputIdentifier not in output:
				executed += recursiveExecute(prompt, output, inputIdentifier, data)
	
	allData = inputData(inputs, objectClass, output, prompt, data)
	newObject = objectClass()

	output[current] = getattr(newObject, newObject.function)(**allData)
	return executed + [current]

def recursiveQueueExecute(prompt, output, current):
	inputs = prompt[current]["inputs"]

	queue = []

	if current in output:
		return []

	for i in inputs:
		data = inputs[i]

		if isinstance(data, list):
			inputIdentifier = data[0]
			outputIndex = data[1]

			if inputIdentifier not in output:
				queue = recursiveQueueExecute(prompt, output, inputIdentifier)

	return queue + [current]



def recursiveDelete(prompt, previous, output, current):
	inputs = prompt[current]["inputs"]
	classType = prompt[current]["classType"]
	objectClass = modules.moduleMap[classType]

	changedPrevious = ""
	changed = ""

	if hasattr(objectClass, "changed"):
		if "changed" not in prompt[current]:
			if current in previous and "changed" in previous[current]:
				changedPrevious = previous[current]["changed"]
			
			allData = inputData(inputs, objectClass)

			changed = objectClass.changed(**allData)
			prompt[current]["changed"] = changed
		else:
			changed = prompt[current]["changed"]

	if current not in output:
		return True

	toDelete = False

	if changed != changedPrevious:
		toDelete = True
	elif current not in previous:
		toDelete = True
	elif inputs == previous[current]["inputs"]:
		for i in inputs:
			data = inputs[i]

			if isinstance(data, list):
				inputIdentifier = data[0]
				outputIndex = data[1]

				if inputIdentifier in output:
					toDelete = recursiveDelete(prompt, previous, output, inputIdentifier)
				else:
					toDelete = True

				if toDelete:
					break
	else:
		toDelete = True

	if toDelete:
		print(f"[!] Deleted {current}")

		deleteThis = output.pop(current)
		del deleteThis

	return toDelete


def validateInput(prompt, item):
	inputs = prompt[i]["inputs"]
	classType = prompt[item]["classType"]
	objectClass = modules.moduleMap[classType]

	inputsClass = objectClass.inputType()

	required = inputsClass["required"]

	for i in required:
		if i not in inputs:
			return (False, f"[!] Missing input: {classType}, {i}")

		value = inputs[i]
		info = required[i]
		inputType = info[0]

		if isinstance(value, list):
			if len(value) != 2:
				return (False, f"[!] Bad input: {classType}, {i}")

			objectIdentifier = value[0]
			objectClassType = prompt[objectIdentifier]["classType"]

			returnType = modules.moduleMap[objectClassType].returnType

			if r[value[1]] != inputType:
				return (False, f"[!] Return type mismatch: {classType}{i}")

			if not r[0]:
				return validateInput(prompt, objectIdentifier)

		else:
			if inputType == "INT":
				inputs[i] = int(value)
			elif inputType == "FLOAT":
				inputs[i] = flloat(value)
			elif inputType == "STRING":
				inputs[i] = str(value)

			if len(info) > 1:
				if "min" in info[1] and value < info[1]["min"]:
					return (False, f"[!] Value smaller than minimum: {classType}, {i}")

				if "max" in info[1] and value > info[1]["max"]:
					return (False, f"[!] Value larger than maximum: {classType}, {i}")					

			if isinstance(inputType, list):
				if val not in inputType:
					return (False, f"[!] Value not in list: {classType}{i}")

	return (True, "")

def validatePrompt(prompt):
	output = set()

	for i in prompt:
		class_ = modules.moduleMap[prompt[i]["classType"]]

		if hasattr(class_, "outputNode") and class_.outputNode == True:
			output.add(i)

	if len(output) == 0:
		return (False, "[!] Prompt has no output")

	good = set()

	for i in output:
		valid = False
		reason = ""

		try:
			validate = validateInput(prompt, i)
			valid = validate[0]
			reason = validate[1]
		except:
			valid = False
			reason = "Parsing error"

		if valid:
			good.add(i)
		else:
			print(f"[!] Failed to validate prompt for output {i}, {reason}")
			print("[!] Output will be ignored")

	if len(good) < 1:
		return (False, "[!] Prompt doesn't have properly connected output")

	return (True, "")

class Executor:
	def __init__(self):
		self.output = {}
		self.previous = {}

	def execute(self, prompt, data={}):
		with torch.no_grad():
			for i in prompt:
				recursiveDelete(prompt, self.previous, self.output, i)

			current = set(self.output.keys())

			executed = []

			try:
				queue = []
				for i in prompt:
					class_ = modules.moduleMap[prompt[i]["classType"]]

					if hasattr(class_, "outputNode"):
						queue += [(0, x)]

				while len(queue) > 0:
					queue = sorted(list(map(lambda a: (len(recursiveQueueExecute(prompt, self.output, a[-1])), a[-1]), queue)))
					i = queue.pop(0)[-1]

					class_ = modules.moduleMap[prompt[i]["classType"]]

					if hasattr(class_, "outputNode")
						if class_.outputNode == True:
							valid = False

							try:
								validate = validateInput(prompt, i)
								valid = validate[0]
							except:
								valid = False

							if valid:
								executed += recursiveExecute(prompt, self.output, i, data)
			except Exception as e:
				print(traceback.format_exc())

				toDelete = []

				for output in self.output:
					if output not in current:
						toDelete += [output]

						if output in self.previous:
							deleteThis = self.previous.pop(output)
							del deleteThis


				for output in toDelete:
					deleteThis = self.output.pop(o)
					del deleteThis

			else:
				executed = set(executed)

				for i in executed:
					self.previous[i] = copy.deepcopy(prompt[i])


def worker(process):
	executor = Executor()

	while True:
		item = process.get()
		executor.execute(item[-2], item[-1])
		process.done()

class PromptServer(BaseHTTPRequestHandler):
	def _set_headers(self, code=200, ct='text/html'):
		self.send_response(code)
		self.send_header('Content-type', ct)
		self.end_headers()

	def log_message(self, format, *args):
		pass

	def do_GET(self):
		if self.path == "/prompt":
			self._set_headers(ct='application/json')
			prompt_info = {}
			exec_info = {}
			exec_info['queue_remaining'] = self.server.process.unfinished_tasks
			prompt_info['exec_info'] = exec_info
			self.wfile.write(json.dumps(prompt_info).encode('utf-8'))
		elif self.path == "/object_info":
			self._set_headers(ct='application/json')
			out = {}

			for i in modules.moduleMap:
				objectClass = nodes.moduleMap[i]
				info = {}
				info['input'] = objectClass.inputType()
				info['output'] = objectClass.returnType
				info['name'] = x #TODO
				info['description'] = ''
				out[i] = info
			self.wfile.write(json.dumps(out).encode('utf-8'))

		elif self.path[1:] in os.listdir(self.server.serverDirectory):
			self._set_headers()
			with open(os.path.join(self.server.serverDirectory, self.path[1:]), "rb") as f:
				self.wfile.write(f.read())
		else:
			self._set_headers()
			with open(os.path.join(self.server.serverDirectory, "index.html"), "rb") as f:
				self.wfile.write(f.read())

	def do_HEAD(self):
		self._set_headers()

	def do_POST(self):
		resp_code = 200
		out_string = ""
		if self.path == "/prompt":
			print("[!] Prompt received")
			self.data_string = self.rfile.read(int(self.headers['Content-Length']))
			json_data = json.loads(self.data_string)

			if "number" in json_data:
				number = float(json_data['number'])
			else:
				number = self.server.number
				self.server.number += 1

			if "prompt" in json_data:
				prompt = json_data["prompt"]
				valid = validatePrompt(prompt)
				extra_data = {}

				if "data" in json_data:
					data = json_data["data"]

				if valid[0]:
					self.server.process.put((number, id(prompt), prompt, data))
				else:
					resp_code = 400
					out_string = valid[1]
					print("[!] Invalid prompt:", valid[1])
		self._set_headers(code=resp_code)
		self.end_headers()
		self.wfile.write(out_string.encode('utf8'))
		return

def run(process, address="127.0.0.1", port=8585):
	serverAddress = (address, port)
	httpd = HTTPServer(serverAddress, PromptServer)
	httpd.serverDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "server")
	httpd.process = process
	httpd.number = 0

	print("[!] Starting server\n")
	print("To see the GUI go to: http://{}:{}".format(serverAddress[0], serverAddress[1]))
	httpd.serve_forever()


if __name__ == "__main__":
	process = queue.PriorityQueue()
	threading.Thread(target=worker, daemon=True, args=(process,)).start()
	run(process, address="127.0.0.1", port=8585)
