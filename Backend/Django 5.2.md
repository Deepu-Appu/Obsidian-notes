First step toward starting Django is creating a directory and creating the environment in it so, that we can install the programs in the environment

first step → creating directory → creating an environment → downloading Django

```Python
# creating new environment
python -m env env # naming environment as env
# In windows use 'dir' and change the directory to ennv\script\ then type
activate
# it will activate the environment then
pip install django --pre
```

###   🔍 ==What== `--pre` ==Does==

By default, `pip` installs only stable version of packages.

The `--pre` flag tells pip to **include pre-release version** when searching for the latest version

### 📁 ==Why Use a== `src` ==Folder in Django Projects== 

Creating a `src` folder in Django project is a common best practice to keep the project structure clean and scalable. It helps avoid clutter in the root directory by grouping all Django-related code (like the project setting and apps) in one place. **It also prevents import conflicts with built-in Python modules** (e.g., naming an app email won’t interfere with the standard email module), improves compatibility with tools like **pytest** , **Docker**, and **tox**, and <b> mirrors how large production-ready projects are organized</b>. While Django doesn’t require it, using a src folder leads to <b>cleaner more maintainable code</b>, especially in growing or collaborative projects.


### ==Start Django server==

In order to start project, we have to create a new Django project name cfehome in the <b>current directory</b>  (because of the dot at the end)

### 🔍 ==What Each Part Means==

- <font color='#CF9FFF'>django-admin</font>   Django’s command-line tool.
- <font color='#CF9FFF'>startproject cfehome </font>  Create a new project named cfehome

```Python

# Inside src folder
django-admin startproject cfehome .
# after creating project name
python manage.py runserver # to run server name
```

###   🔐 ==What is `SECRET_KEY`?==

In Django, the `SECRET_KEY` is a critical settings used for:

- Cryptographic signing (cookies, sessions, tokens)
- Password reset links
- Hashing algorithms

If it’s exposed or predictable, it could lead to security vulnerabilities.

```Python
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
```

###   ✅ ==Why use==  
`os.environ.get()`?

We are using this way, so that the key is set **outside the codebase**, typically via:

- Terminal
- .env file (with python-dotenv or django-environ)
- Hosting platform config (e.g., Heroku, Vercel, etc.)

###   🔍  `os.environ.get()` ==in Python==

The method `os.environ.get()` is used to safely access environment variables in your Python code.

### ==Use gh cli (which is nothing but git cli)==

- First of all install ghcli
- Then `gh auth`
- login through git → Https → browser → login (by giving code and permission)

```Python
git remote -v  # to check wheather there is branch or not
```

###   🔍 ==What is== `origin` ==in Git?==

- Origin is an alias for the **remote Git URL** (like GitHub, GitLab, etc.)
- It makes it easier to refer to the remote repository without typing the full URL every time.

```Python
# If we are in branch called master , not main
git push origin master
# We can check by using command 
git branch
# we can rollback, first we have to check git log
git log
# then 
git reset "paste id" --hard
# to check status 
git status
# find out the change
git diff
# forcing 
git push  --force
```

### ==Git Action==

GitHub Actions is a tool that helps you automate tasks whenever something happens in your GitHub repo — like pushing code, opening a pull request, or releasing a version

go to → action → django → then change python version according to your choice → commit it (django.yml file you will get)

if you are in master:

`git branch --set-upstream-to=origin/master master`

then:

`git pull` (push <span style="color:rgb(250, 128, 114)">django.yml</span> file)

then after you have to create a <span style="color:rgb(250, 128, 114)">requirement.txt</span> file

with latest django version like: `Django==5.2.4`

then, `git add --all` → `git commit -m “Added requirement.txt”` → then add the path of [manage.py](http://manage.py) in yml file

###   🗂️ ==What is==  `.gitkeep` ==in Git?==

<font color ='#CF9FFF'>.gitkeep</font> is not a **official Git feature**, but a **commonly used convention**.

###   ✅ ==Purpose of==  `.gitkeep`

Git **does not track empty folder**. If a folder has no files, Git will ignore it completely

So **developers add a dummy file** called .gitkeep to force Git to:

- Track the folder
- Commit it into the repository

  

```Python
DEBUG = os.environ.get("DEBUG") == "true"
```

This is a **common Django pattern** used to control whether the app runs in **debug mode** based on environment variable.

### 🔍 ==Explanation:==

- `os.environ.get(”DEBUG”)` :
    
    Reads the value of an environment variable named `DEBUG`
    
- `== “true”`:
    
    Compares it to the string `“true” (case-sensitive)`
    

so in order to start the project we have to give command

`$env:DEBUG="true"` → `python` `manage.py runserver`

then we have to give or set secret key:

`$env:DJANGO_SECRET_KEY="your-very-secret-key"` → `python` `manage.py runserver`
 

We store values like `SECRET_KEY` and `DEBUG` in a `.env` file to separate sensitive **configuration from your source code**

```YAML
from dotenv import load_dotenv

load_dotenv()  # take environment variables
```

### ✅ ==What this does:==

1. `load_dotenv()` read the .env file from your project’s root directory.
2. It loads all variables inside it into your environment so you can access them using :
    
    ```Python
    # for secret key
    SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
    # inorder to enable debug
    DEBUG = os.environ.get("DJANGO_DEBUG") == "true"
    ```
    

###   ✅ ==What is== `python-decouple`?

`python-decouple` helps you manage **environment variables** and sensitive settings like `SECRET_KEY` , `DEBUG, or database credentials without hardcoding them into your codebase`.

```Python
from decouple import config
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config('DJANGO_SECRET_KEY',cast=str, default=get_random_secret_key())
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config("DJANGO_DEBUG", cast=bool, default=False)
```

### 🔐 `ALLOWED_HOSTS`

```Python
ALLOWED_HOSTS = [
    # "first-domain.page.gd",
    # "www.first-domain.page.gd",
    ".railway.app",
]
```

### ✅ ==What this does:==

- Tells Django to accept request from specific hostnames.
- The “`.railway.app`” format (with leading do) allows subdomains, like
    - `yourapp.railway.app
    - `project-name.up.railway.app

### 🔒 `CSRF_TRUSTED_ORIGIN`

```Python

CSRF_TRUSTED_ORIGINS = [
    "https://*.railway.app",
]
```

### ✅ ==What this does:==

- Allows requests with CSRF tokens from specific trusted domains
- Wildcard() do work here, so `.railway.app` is okay

  

Then in [setting.py](http://setting.py) under base directory

```Python
# create a base directory 
REPO_DIR = BASE_DIR.parent
# create a template directory
TEMPLATES_DIR = BASE_DIR / "templates"
# if the template directory is not there create one
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
```

Inside the template directory create hello-world.html file then create file name called **view**

```Python
 from django.http import HttpResponse
from django.shortcuts import render

def hello_world(request):
    return render(request, "hello-world.html", {})




def healthz_view(request):
    return HttpResponse("ok")
```

### ✅ ==What's happening here?==

You’re **importing tools** from Django to help you:

- HttpRespone lets you send plain text back to a web browser
- render lets you send an <b>HTML file</b> (webpage) back to a user

### 📄==Function 1:== `hello_world`

- This is a **view function.**
- when someone visits a certain URL (like `/hello/` ), this function is called
- Django will:
    1. Take the `request (the browser asking for a page)`
    2. Look for an `HTML file called hello-world.html`
    3. Show that page to the user
    4. The `{} means you’re not sending any extra data to the HTML for now`

### 📄 ==Function 2:== `healthz_view`

- This is another simple view.
- It just return **the text** “**ok**” — nothing fancy.
- It’s usually used as a **health check endpoint**.
    - For example, Railway or a cloud platform might visit this URL to check if your app is alive and responding

### 🧠 ==What is== `urlpatterns`?

In Django, `urlpatterns` is a list that tells Django:

> “Hey, when someone visits a certain URL, run this function (called a view)”

It’s like a **map** that connects your website addresses (URLs) to **Python functions** that decide what to show on the screen  
  

```Python
urlpatterns = [
    path('', views.hello_world),
    path('healthz/', views.healthz_view),
    path('admin/', admin.site.urls),
]
```

This will return the `hello-world.html` , `healthz` from the view

### ==What is WSGI? (Web Server Gateway Interface)==

Think of wsgi as a “translator” between your python code and web server.

### ==The Problem Without WSGI:==

<span style="color:rgb(250, 128, 114)">Imagine you have</span>:

- A **web server** (like Apache or Nginx) → speaks “web language”
- **Your Django app** (python code) → speaks “Python language”

They can’t talk to each other directly! It’s like trying to have a conversation between someone who only speaks English and someone who only speaks Spanish.

Web Server ←→ WSGI ←→ Your Django App

**WSGI** acts as the translator that:

- Takes requests from the web server
- Converts them into something your Python code understands
- Takes your Python response
- Converts it back to something the web server can send to browsers

### ==Why Waitress?==

waitress is a production-ready WSGI server that

- <span style="color:rgb(250, 128, 114)">is fast</span> (handles multiple requests simultaneously)
- <span style="color:rgb(250, 128, 114)">is secure</span> for production use
- <span style="color:rgb(250, 128, 114)">won’t crash</span> under normal load
- <span style="color:rgb(250, 128, 114)">works great on windows</span> (unlike some other servers)

### ==The Command Breakdown==

```Python
waitress-serve --listen=0.0.0.0:8000 cfehome.wsgi:application
```

- `waitress-serve` : “Hey waitress, start serving my app”
- `--listen=0.0.0.0:8000` : “Listen for requests on port 8000”
- `cfehome.wsgi:application` : “Use the WSGI application from this file”

### ==Deployment==

For deployment we use railway configure your GitHub. then select the project we need to deploy use `cd src && gunicorn cfehome.wsgi:application --bind 0.0.0.0:8000` (setting → deployment) we have to use gunicorn instead of waitress in railway. Configure the (setting → network) network port from `8080` to `8000`.`(If any error happen check the logs)`

```Python
PROJECT_NAME = config("PROJECT_NAME", default="Unset Project Name")
```

From [settings.py](http://settings.py) import settings → print the project name in hello_world function

```Python
def hello_world(request):
    return render(request, "hello-world.html", {
        "project_name": PROJECT_NAME 
    })
```

`{ "project_name": PROJECT_NAME }` :

- This dictionary is the context passed to the template.
- Inside `hello-world.html` , you can now use the variable `{{ project_name }}` to display the value of the PROJECT_NAME

### ==How to generate secret key==

```Python
from django.core.management.utils import get_random_secret_key
print(get_random_secret_key())
```

after getting the key store it in .env file and get access of it using decouple module → config

### ✅ ==What is Django ORM?==

**Django ORM (Object-Relational Mapping)** is a built-in feature in Django that allows you to interact with your **database** using **Python classes** and **object** , rather **than** writing raw SQL queries.

### ✅ ==What it does==

This runs all pending database migrations without prompting for user input (like confirmations)

- `—no-input` : Prevents Django from asking **interactive questions**, which is useful when running in **automated environments**.

In deployment (railway) → add the command pre-production.

### ==Create a Database==

Creating a database using postgressql in the railway, then in variable copy `DATABASE_PUBLIC_URL` and paste it in .env file `DATABASSE_URL=postgresql://postgres:lIdFfrTzqqMOMtwPcyoSPBlXkxHvRZjJ@tramway.proxy.rlwy.net:36164/railway`  
then install dj-database-url and psycopg[binary]  

###   🔧  `dj-database-url` — ==What It Is and How to Use It==

`dj-database-url` is a helper library in Django projects that lets you configure your database setting using a single DATABASE_URL string, commonly used in `.env` files or cloud platform like **Railway**, **Heroku**, **etc**

Imagine you’re setting up your Django project and you want it to connect to a database. Normally, you’d have to fill in a bunch of technical details like:

- Database name
- Username
- Password
- Host
- Port

It’s a bit like filling out a form with different fields.

👉`dj-database-url` simplifies that by letting you provide just one line called a **database URL**

###   🐘  `psycopg[binary]` — ==What It Is and How to Use It==

`psycopg[binary]` is the **binary wheel variant** of `psycopg` — a PostgresSQL adapter for Python. It’s the **recommended** way to install **psycopg** when you don’t want to compile anything from source.

Django speaks **Python**, but your database (like PostgreSQL) speaks **SQL**

So you need a **translator** between them.

That translator is `psycopg`.  
  

![[image.png|image.png]]

1. <span style="color:rgb(250, 128, 114)">Postgres-dev</span>:
    - A development database (not for the public, used for testing)
2. <span style="color:rgb(250, 128, 114)">Postgres-Prod</span>:
    - The main database used by your real (production) app.

We have to delete the postgres-dev by clicking on to it and then → setting → terminate at the bottom.

clear the `database url` in the `env`.

Then create a variable in django for the postgres-prod database →

![[/image 1.png|image 1.png]]

In the `VARIABLE_NAME` → `DATABASE_URL` check the drop down you will see the database url of the `database-prod`

select it and deploy it.

Remove the public network in the postgres-prod in setting. (we want to use private network).

### 🧱 ==What is a Private Network in Railway?==

In Railway, a **private network** connects your services (like your Django app and PostgreSQL database) **internally**, without going over the public internet.

Think of it like a **secure private room** where your app and database can talk without anyone outside listening in.

### Superuser with Custom Django Management Command

In order to create a login interface we first give command to create app

`python` `[manage.py](http://manage.py)` `startapp commando`

then create a python file in this directory:

`src/commando/management/commands/auto_admin.py`

then create `__init___.py` in the `commands directory` 