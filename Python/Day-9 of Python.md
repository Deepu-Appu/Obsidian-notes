### ==Dictionaries==
The `Dictinary` are data structure of Python programming, It has `key` and `value` pair, the format is like `{key:value}`. For example `key` is `bug` and the `value` will be `I saw three bugs in my garden` , So it can be displayed as `{"bug": "I saw three bugs in my garden"}` . You can add more `key` `value` by adding `,` at the end.

like:
`programming_dictionary = {
`"Apple": "two"
`"Banana": "one"
`"Grape": "four"
`}`

- In order to access the value in the dictionary, we access like this `print(programming_dictionary["Apple"])` using the key.
- To add `value` to the dictionary we create a new `key` and assign `value` to it.
	`programming_dictionary["Loop"] = "The action of doing something over and over again"`
- We can edit the `value` in the same way.
If we for loop the dictionary we will get the `key` as output so in order to get `values` we have to:
```python
for key in programming_dictionary:
	print(key) # this will print key
	print(programming_dictionary[key]) #this prints the values
```

### ==Exercise==

You have access to a database of `student_scores` in the format of a dictionary. The **keys** in `student_scores` are the names of the students and the values are their exam scores. 
- Write a program that **converts their scores to grades**.
- By the end of your program, you should have a new dictionary called `student_grades` that should contain student names as **keys** and their assessed grades for **values**. 
- The final version of the `student_grades` dictionary will be checked. 
- **DO NOT** modify lines 1-7 to change the existing `student_scores` dictionary. 
- This is the scoring criteria: 
	- Scores 91 - 100: Grade = "Outstanding" 
	- Scores 81 - 90: Grade = "Exceeds Expectations" 
	- Scores 71 - 80: Grade = "Acceptable" 
	- Scores 70 or lower: Grade = "Fail"
	
```python
student_scores = {
    'Harry': 88,
    'Ron': 78,
    'Hermione': 95,
    'Draco': 75,
    'Neville': 60
}

# function for returning the class grade
def scores_calc(scores, grades):
    for name, score in student_scores.items():
        if (score >= 91) & (score <= 100):
            student_grades[name] = "Outstanding"
        elif (score >= 81) & (score <= 90):
            student_grades[name] = "Exceeds Expectations"
        elif (score >= 71) & (score <= 80):
            student_grades[name] = "Acceptable"
        elif (score <= 70):
            student_grades[name] = "Fail"
    
    print(student_grades)

# created empty dictionary
student_grades = {}
# calling the function
scores_calc(student_scores, student_grades)
```

- we can also add list in dictionary:

```python
travel_log = {
"france": ["Paris", "Lillie", "Dijon"],
"Germany": ["Stuttgart", "Berlin"]
}
```

- We can access the list like:
	`print(travel_log["France"][1])` 
	- here we are accessing the element `"Lillie"` in the list which is in the dictionary , first we accessing the `key:"france"` and then we need `"Lilie"` which is in the $1^{st}$ index of the list.
- We can also nest dictionary inside dictionary:

```python 
travel_log = {
"france": {
	"cities_visited": ["Paris", "Lille", "Dijon"],
	"total_visits":12
},
"germany":{
	"cities_visited": ["Berlin", "Hamburg", "Stuttgart"],
	"total_visits": 5
}
}
```

- To print `"Stuttgart"` we have to get the element inside the `"germany" -> "cities_visited" -> "stuttgart"` that is:
	`print(travel_log["Germany"]["cities_visited"][2])`