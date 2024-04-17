from typing import List


def multiplicate(arr: List[int]) -> List[int]:
	"""
	Return a list of 'multiplications'.
	
	1. Calculate product of all elements in 'arr'.
	2. For each element in 'arr':
		Divide the total product onto this element and append result.
	3. Return list of resulted values.

	Args:
		arr: List[int] - non-empty input list of integers.
	Returns:
		res: List[int] - output list of 'multiplications'.
	Raises:
		ValueError - if either of the following:
			1. 'arr' is empty.
			2. 'arr' integer, that is less or equal 0.
		TypeError - if either of the following:
			1. 'arr' is not iterable.
			2. 'arr' contains non-integers.
			3. 'arr' has no __len__ method defined.
	"""
	if not len(arr):
		raise ValueError("Input list should not be empty.")
	val = 1
	res = []
	for elem in arr:
		if not isinstance(elem, int):
			raise TypeError("Elements of list should be integers.")
		if elem <= 0:
			raise ValueError("Elements of list should be positive integers.")
		val *= elem
	for elem in arr:
		res.append(val // elem)
	return res


if __name__ == "__main__":
	# Usage example
	arr = [1,2,3,4]
	res = multiplicate(arr)
	print("Input:", arr)
	print("Output:", res)