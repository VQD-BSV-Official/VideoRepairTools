def ransomware(file_path):
	with open(file_path, 'rb') as dump:
		corrupt_data = chunk = dump.read() # Read byte in file

		# //////////////////////////////////////////////////////////
		print('Find atom "moov"...'); offset_atom_moov = -1
		offset_atom_moov = chunk.rfind(b'moov') # Find byte "moov"
		if offset_atom_moov != -1:
			print(f'Found atom "moov" offset: {offset_atom_moov}')

		# //////////////////////////////////////////////////////////
		elif offset_atom_moov == -1: # Note found
			print(f'Atom "moov" not found')
			print("\n!!! Can't repair T_T You can refer to the usage 1?")
			return

	# //////////////////////////////////////////////////////////

	dir_path = os.path.dirname(file_path) # Lấy đường dẫn
	file_ext = os.path.splitext(file_path)[1] # Lấy đuôi
	# Tên file không đuôi
	remove_ext = os.path.splitext(os.path.basename(file_path))[0]

	# //////////////////////////////////////////////////////////
	
	length_to_offset_8 = offset_atom_moov - 4
	length_to_moov_bytes = length_to_offset_8.to_bytes(8, byteorder='big')

	fisrt_16bytes = b'\x00\x00\x00\x01mdat' + length_to_moov_bytes
	with open(f'{dir_path}/{remove_ext}_R{file_ext}', "wb") as w:
		w.write(fisrt_16bytes + corrupt_data[16:])

def run(file_path):
	ransomware(file_path)

import os