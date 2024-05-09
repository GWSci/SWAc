import time
import unittest

class TimedTest:
	def __init__(self, test, time):
		self.test = test
		self.time = time

def main():
	all_tests = discover_tests()
	test_timings = time_tests(all_tests)
	table = format_test_timings(test_timings)
	print(table)

def discover_tests():
	tests = unittest.defaultTestLoader.discover(".")
	result = []
	discover_tests_recursively(result, tests)
	return result

def discover_tests_recursively(acc, tests):
	if hasattr(tests, "__iter__"):
		for t in tests:
			discover_tests_recursively(acc, t)
	else:
		acc.append(tests)

def time_tests(all_tests):
	result = []
	for t in all_tests:
		start_time = time.perf_counter()
		test_result = t.run()
		end_time = time.perf_counter()
		elapsed_time = end_time - start_time
		result.append(TimedTest(t, elapsed_time))
		if (len(test_result.failures) > 0 or len(test_result.errors) > 0):
			print(f"{test_result} {t}")
	return result

def format_test_timings(test_timings):
	result = ""

	for t in sorted(test_timings, key=lambda t: t.time):
		result += f"{t.time:10.4f} s - {t.test}\n"
	
	total_time = 0.0
	for t in test_timings:
		total_time += t.time
	
	result += f"\n"
	result += f"Test Count: {len(test_timings)}\n"
	result += f"Total Time: {total_time:10.4f} s\n"
	return result

if __name__ == "__main__":
	main()
