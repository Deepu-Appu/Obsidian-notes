We can call two function and return the output using the `return`
```python
# First func
def function_1(text):
	return text + text # returns concatinated text

# Second Function 
def function_2(text):
	return text.title() # returns titled text

# storing the output in the "output"
output = function_2(function_1("hello"))
print(output)

```

### ==Docstrings In Python==

Python documentation string (or docstrings) provide a convenient way of associating documentation with python modules, functions, classes, and methods. It's specified in source code that is used, like comment, to document a specific segment of code. Unlike conventional source code comments, the docstring should describe what the function does, not how.

```python
def format_name(f_name. l_name):
	"""Take a first and last name and 
	format it to return  the title case 
	version of the name.""" # this is docstring 
	formated_f_name = f_name.title()
	formated_l_name = l_name.title()
	return f"{formated_f_name} {formated_l_name}"

formatted_name = format_name(f_name: "AnGeLa", l_name: "YU")

length = len(formatted_name) # len is a in built-function that returns the lenght of the words
```

### ==Exercise==

Write a program that returns True or False whether if a given year is a leap year. A normal year has 365 days, leap years have 366, with an extra day in February. The reason why we have leap years is really fascinating, [this video](https://www.youtube.com/watch?v=xX96xng7sAE) does it more justice. This is how you work out whether if a particular year is a leap year:
- on every year that is divisible by 4 with no remainder
- **except** every year that is evenly divisible by 100 with no remainder 
- **unless** the year is also divisible by 400 with no remainder

```python
def is_leap_year(year):
    # Write your code here. 
    # Don't change the function name
    if (year % 4 == 0):
        if (year % 100 == 0):
            if (year % 400 == 0):
                return True
            else:
                return False
        else:
            return True
    else:
        return False
        
is_leap_year(2025)
```
