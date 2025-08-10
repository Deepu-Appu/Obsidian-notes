### ==Function==
We can say function is a **block of code where complex code** have been **stored and  accessed when we needed**, We can call the `functionName()`

```
def my_function():
#Do this 
#Then do this
#Finally do this
```

```python
def greet():
	print("Hello")
	print("How do you do?")
	print("Isn't the weather nice?")
	
greet()
```

==Function with input==

```python
def greet_with_name(name):
	print(f"Hello {name}")
	print(f"How do you do {name}")

greet_with_name("Aman")
```
- We add placeholder for the input, you can say in the function which is really ==parameter==
- Where we assign some input is called the ==argument==

==Exercise==
Create a function called `life_in_weeks()` using math's and f-Strings that tells us how many weeks we have left, if we live until 90 years old.
- It will take your current age as the input and output a message with our time left in this format:
	- `You have x weeks left.`
- Where x is replaced with the actual calculated number of weeks the input age has left until age 90.

```python
def life_in_weeks(age):
    # if we live upto 90 the years we have is 90 - current age
    years = 90 - age
    # one year has 52 weeks, then total week left is: the number of year left x week in one year
    total_weeks_left = 52 * years
    outcome = print (f"You have {total_weeks_left} weeks left.")
    
# calling the function
life_in_weeks(12)
```

`method`: To find how much weeks you have left, if you live till
- First we have to find how many years are left, that is `90 - current age`
- Then we have to find how many weeks are left for for one year we have 52 week, So the total weeks left is `52 * years left`
### ==Multiple parameter==

```python
def greet_with(name, location):
	print(f"hello {name}")
	print(f"What is it like in {location}")

greet_with(name:"jack Bauer", location: "Nowhere")
```

- We are assigning the value with `name: "jack Bauer"` , the `"jack Bauer` is assigned as the `name` parameter
### ==Positional Argument & Keyword Argument==

``` python

def my_function(a, b, c):
#Do this with a 
#Then do this with b
#Finally do this with c

# positional argument
my_function(1, 2, 3)

# keyword argument
my_function(a=2, b=1, c=3)
```

- Here in  `Positional argument` the value is assigned according to the position they are in 
- And the `Keyword argument` the values is directly assigned to the variable they are in (position does not matter)
### ==Exercise==
You are going to write a function called `calculate_love_score()` that tests the compatibility between two names.  To work out the love score between two people: 
-  Take both people's names and check for the number of times the letters in the word TRUE occurs.   
-  Then check for the number of times the letters in the word LOVE occurs.   
- Then combine these numbers to make a 2 digit number and print it out.

```python
def calculate_love_score(name1, name2):
    letter = { 't':0,'r':0,'u':0,'e':0,'l':0,'o':0,'v':0,'e':0}
    combined_name = name1+name2 
    new_combined_name = combined_name.lower()
    for char in new_combined_name:
        for l in letter.keys():
            if l == char:
                letter[l] += 1

    first = ['t','r','u','e']
    second = ['l','o','v','e']
    
    total1 = 0
    total2 = 0

    for k in letter.keys():
        if k in first:
            total1 += letter[k]
        if k in second:
            total2 += letter[k]
    
    Grand_total = str(total1) + str(total2)
    print(Grand_total)
        
calculate_love_score('kanye West','Kim Kardashian')
```

- `def calculate_love_score(name1, name2)`:
	-  first a dictionary is created `letter`  which contains letter of true love as key and value in initialized as 0.
	- first we combine `name1` and `name2` and store it in variable called `combined_name`
	- then we convert it into lower case and assign it into new variable called `new_combined_name`
	- for loop the `new_combined_name` and `letter` dictionary `key` 
	- if they match add 1
	- then create a list for ==true== : `first` and ==love== : second 
	- initialize `total1` and `total2` to store score of `first` and `second variable`
	- for loop letters if `letter` is in `first` add the score from `letter` and same for `second`
	- create a variable `Grand_total` which `str` of `first` and `second` -> we need score of letters`true` and`love`
- At last we call the function `calculate_love_('kanye West','Kim Kardashian')`