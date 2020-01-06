import sys
import numpy as np
import matplotlib.pyplot as plt




'''
Set of functions useful for retireving, so far, flattened arrays from txt files.

Functions Present:
	-str_list_to_np_array(input_str, dtype=np.float32):
		*Function that takes a python list that has been directly transcribed to a string and converts it to a numpy
		 array with the specified data type (default np.float32).

	-get_array_from_txt_w_list(file_path, return_type="np"):
		*Function that takes a flattened python list dirrectly translated to string form from the designated file and
		 returns it as either a (default) numpy array, a python list, or a PyTorch tensor.
'''



def str_list_to_np_array(input_str, dtype=np.float32):
	return np.fromstring(input_file_string.strip("[").strip("]"), sep=", ", dtype=dtype)




def get_array_from_txt_w_list(file_path, return_type="np"):

	input_file = open(file_path, "r")

	input_file_string = ""
	for line in input_file:
		input_file_string = input_file_string + line

	input_array = str_list_to_np_array(input_file_string)

	if return_type == "np":
		return input_array
	elif return_type == "list":
		return input_array.tolist()
	elif return_type == "float_tensor":	
		try:
			import torch
			print("Successfuly imported torch.")
			return torch.FloatTensor(input_array)
		except ImportError:
			print("Could not import torch. Returning numpy array so you can save it and try again yourself.")
			return input_array
		except Exception as e:
			print("Imported torch successfully, but for some reason could not turn your array from numy to FloatTensor. Here's the error message:\n\n" + str(e) + "\n\nReturning numpy array.")
			return input_array
	else:
		print("No recognized data type was given (", return_type, "), returning a numpy array.")
		return input_array




def main():
	argv = sys.argv
	argc = len(sys.argv)
	output_file_path = ""

	if argc == 4:
		print("Assuming the first command line argument is the path to the file to access, the second is the return type of the obtained array, and the last is the name of the output file to write the obtained array to.")
		input_file_path  = argv[1]
		return_type 	 = argv[2]
		output_file_path = argv[3]
		output_array 	 = get_array_from_txt_w_list(input_file_path, return_type)
	elif argc == 2:
		print("I think you just enetered a filename and no return array type, so assuming your command line argument is a path and returning a numpy array.")
		input_file_path = argv[1]
		output_array = get_array_from_txt_w_list(input_file_path)
	elif argc == 3:
		print("Assuming first command line argument is the path to the file you want to access and that the second is the type of the array to return.")
		input_file_path = argv[1]
		return_type 	= argv[2]
		output_array = get_array_from_txt_w_list(input_file_path, return_type)
	else:
		print("Either none or too many command line arguments were given and not sure what to do.")
		print("Usage:\n\tpython3 get_array_from_txt.py input_file_path [return_type] [output_file]")
		print("\t-input_file_path: string indicating the path to the file to access.")
		print("\t-[return_type]: string indicating the type of array to return. Options are:")
		print("\t\t-np: numpy array.\n\t\t-list: python list.\n\t\t-float_tensor: will try to return Pytorch FloatTensor object. If it can't, will default to numpy array.")
		
	if output_file_path == "":
		print("No output file was specified/obtained, using default name: obtained_array.txt")
		output_file_path = "obtained_array.txt"

	output_file = open(output_file_path, "w")
	if str(type(output_array))[8:13] == 'numpy':
		print("Manually increasing length of numpy print option to length of obtained array to avoid truncation.")
		np.set_printoptions(threshold=len(output_array))
		output_file.write(str(output_array).replace("\n", ""))
		print("Resetting numpy print options to default.")
		np.set_printoptions(threshold=1000)
	elif str(type(output_array))[8:13] == 'torch':
		print("Manually increasing length of Pytorch print option to length of obtained array to avoid truncation.")
		torch.set_printoptions(profile="full")
		output_file.write(str(output_array).replace("\n       ", ""))
		print("Resetting Pytorch print options to default.")
		torch.set_printoptions(profile="default")
	else:
		output_file.write(str(output_array))

	output_file.close()

if __name__ == "__main__":
	main()