from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, FloatField, validators
from passlib.hash import sha256_crypt
from functools import wraps

app = Flask(__name__)

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysqlwhy'
app.config['MYSQL_DB'] = 'eateasyapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# init MYSQL
mysql = MySQL(app)

# Index
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = RestaurantSearchForm(request.form)
    if request.method == 'POST' and form.validate():
        # Get Form Fields
        restaurant = request.form['restaurant']
        likes = request.form['likes']
        allergies = request.form['allergies']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get menu items after apply filters
        sql_query = (
            "SELECT s.menu_item, s.description, s.price, r.rec_rating "
            "FROM r_table AS r, s_table AS s "
            "WHERE s.rest_name = %(restaurant)s "
            "AND s.id = r.id "
            "AND s.menu_item = r.menu_item "
            "AND s.description LIKE %(likes)s "
            "AND s.description NOT LIKE %(allergies)s "
            "ORDER BY r.rec_rating DESC"
        )

        results = cur.execute(sql_query, {'restaurant': restaurant, 'likes': '%'+likes+'%', 'allergies': '%'+allergies+'%'})

        data = cur.fetchall()
        # Close connection
        #cur.close()

        if results > 0:
            return render_template('results.html', restaurant=restaurant, results=data)
        else:
            msg = 'No Matching Restaurant Found'
            return render_template('results.html', msg=msg)
    return render_template('search.html')

# Restaurant Search Form Class
class RestaurantSearchForm(Form):
    restaurant = StringField('Restaurant', [validators.Length(min=1, max=50)])
    likes = StringField('Likes', [validators.Length(min=2, max=25)])
    allergies = StringField('Allergies', [validators.Length(min=2, max=25)])

# Restaurants
@app.route('/restaurants')
def restaurants():
    # Create cursor
    cur = mysql.connection.cursor()

    # Get restaurants
    result = cur.execute("SELECT DISTINCT(rest_name) FROM s_table")

    restaurants = cur.fetchall()

    if result > 0:
        return render_template('restaurants.html', restaurants=restaurants)
    else:
        msg = 'No Restaurants Found'
        return render_template('restaurants.html', msg=msg)
    # Close connection
    cur.close()

#Single Restaurant
@app.route('/restaurant/<string:name>/')
def restaurant(name):
    # Create cursor
    cur = mysql.connection.cursor()

    sql_query = (
        "SELECT s.menu_item, s.description, s.price, r.rec_rating "
        "FROM r_table AS r, s_table AS s "
        "WHERE s.rest_name = %(name)s "
        "AND s.id = r.id "
        "AND s.menu_item = r.menu_item "
        "ORDER BY r.rec_rating DESC"
    )
    # Get menu items
    result = cur.execute(sql_query, {'name': name})

    menu_items = cur.fetchall()

    return render_template('restaurant.html', restaurant=name, results=menu_items)

# About
@app.route('/about')
def about():
    return render_template('about.html')

# Register Form Class
class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)", (name, email, username, password))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('You are now registered and can log in', 'success')

        return redirect(url_for('login'))
    return render_template('register.html', form=form)

# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username

                flash('You are now logged in', 'success')
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')

# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))

# Dashboard
@app.route('/dashboard')
@is_logged_in
def dashboard():
    # Create cursor
    cur = mysql.connection.cursor()
    
    sql_query = (
        "SELECT s.rest_name, s.menu_item, s.description, s.price, r.rec_rating, u.user_rating, u.user_comments "
        "FROM r_table AS r, s_table AS s, user_reviews as u "
        "WHERE s.id = r.id "
        "AND s.rest_name = u.rest_name "
        "AND s.menu_item = r.menu_item "
        "AND s.menu_item = u.menu_item "
        "AND u.username = %(username)s"
        "ORDER BY u.user_rating, r.rec_rating DESC"
    )

    result = cur.execute(sql_query, {'username': session['username']})

    results = cur.fetchall()

    if result > 0:
        return render_template('dashboard.html', results=results)
    else:
        msg = 'No Saved Menu Items Found'
        return render_template('dashboard.html', msg=msg)
    # Close connection
    cur.close()

# Review Form Class
class ReviewForm(Form):
    user_rating = FloatField('My Rating')
    user_comments = TextAreaField('Comments')

# Add Review
@app.route('/add_review/<string:restaurant>/<string:menu_item>', methods=['GET', 'POST'])
@is_logged_in
def add_review(restaurant, menu_item):
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        user_rating = form.user_rating.data
        user_comments = form.user_comments.data

        # Create Cursor
        cur = mysql.connection.cursor()

        # Execute
        cur.execute("INSERT INTO user_reviews VALUES(%s, %s, %s, %s, %s)",(session['username'], restaurant, menu_item, user_rating, user_comments))
        # Commit to DB
        mysql.connection.commit()

        #Close connection
        cur.close()

        flash('Review Created', 'success')

        return redirect(url_for('dashboard'))

    return render_template('add_review.html', form=form)


# Edit Review
@app.route('/edit_review/<string:restaurant>/<string:menu_item>', methods=['GET', 'POST'])
@is_logged_in
def edit_review(restaurant, menu_item):
    # Create cursor
    cur = mysql.connection.cursor()

    # Get review
    result = cur.execute("SELECT * FROM user_reviews WHERE username = %s AND rest_name = %s AND menu_item = %s", [session['username'], restaurant, menu_item])

    single_review = cur.fetchone()
    cur.close()
    # Get form
    form = ReviewForm(request.form)

    # Populate review form fields
    form.user_rating.data = single_review['user_rating']
    form.user_comments.data = single_review['user_comments']

    if request.method == 'POST' and form.validate():
        user_rating = request.form['user_rating']
        user_comments = request.form['user_comments']

        # Create Cursor
        cur = mysql.connection.cursor()
        app.logger.info(restaurant, menu_item)
        # Execute
        cur.execute ("UPDATE user_reviews SET user_rating=%s, user_comments=%s WHERE username=%s AND rest_name=%s AND menu_item=%s",(user_rating, user_comments, session['username'], restaurant, menu_item))
        # Commit to DB
        mysql.connection.commit()

        #Close connection
        cur.close()

        flash('Review Updated', 'success')

        return redirect(url_for('dashboard'))

    return render_template('edit_review.html', form=form)


@app.route('/delete_review/<string:restaurant>/<string:menu_item>', methods=['POST'])
@is_logged_in
def delete_review(restaurant, menu_item):
    # Create cursor
    cur = mysql.connection.cursor()

    # Execute
    cur.execute("DELETE FROM user_reviews WHERE username = %s AND rest_name = %s AND menu_item = %s", [session['username'], restaurant, menu_item])

    # Commit to DB
    mysql.connection.commit()

    #Close connection
    cur.close()

    flash('Menu Item Deleted', 'success')

    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.secret_key='secret123'
    app.run(debug=True)
