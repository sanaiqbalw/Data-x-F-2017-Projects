use eateasyapp;

-- DROP TABLE IF EXISTS users;

-- CREATE TABLE users (
-- 	id INT(11) AUTO_INCREMENT PRIMARY KEY,
-- 	name VARCHAR(100),
--  email VARCHAR(100),
--  username VARCHAR(30),
--    password VARCHAR(100),
--    register_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

DROP TABLE IF EXISTS user_reviews;

CREATE TABLE user_reviews (
	username VARCHAR(30),
	rest_name VARCHAR(30) NOT NULL,
	menu_item VARCHAR(30) NOT NULL,
	user_rating FLOAT(30),
    user_comments VARCHAR(200)
);


DROP TABLE IF EXISTS r_table, s_table;

CREATE TABLE s_table (
	id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
	business_id INT(6) UNSIGNED,
	rest_name VARCHAR(30) NOT NULL,
	menu_item VARCHAR(30) NOT NULL,
	description VARCHAR(50),
    price FLOAT(30)
);

CREATE TABLE r_table (
	id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
	business_id INT(6) UNSIGNED,
	menu_item VARCHAR(30) NOT NULL,
	rec_rating FLOAT(30)
);

INSERT INTO s_table (`business_id`,`rest_name`,`menu_item`,`description`, `price`)
VALUES (1, 'HoC', 'lamb briyani', 'very yummy, meaty', 10.91);
INSERT INTO s_table (`business_id`,`rest_name`,`menu_item`,`description`, `price`)
VALUES (3, 'McDonalds', 'fries', 'golden, fried', 3.50);
INSERT INTO s_table (`business_id`,`rest_name`,`menu_item`,`description`, `price`)
VALUES (1, 'HoC', 'mango lassi', 'made of mangoes and yogurt', 4.25);



INSERT INTO r_table (`business_id`,`menu_item`,`rec_rating`)
VALUES (1, 'lamb briyani', 99.0);
INSERT INTO r_table (`business_id`,`menu_item`,`rec_rating`)
VALUES (7, 'ice cream', 50.0);
INSERT INTO r_table (`business_id`,`menu_item`,`rec_rating`)
VALUES (1, 'mango lassi', 90.0);

-- select s.business_id, s.rest_name, s.menu_item, s.description from eateasyapp.s_table as s where s.rest_name='HoC';
SELECT s.menu_item, s.description, s.price, r.rec_rating
FROM eateasyapp.r_table AS r, eateasyapp.s_table AS s
WHERE s.rest_name = 'HoC'
	AND s.business_id = r.business_id
	AND s.menu_item = r.menu_item
    AND s.description LIKE '%yogurt%'
	AND s.description NOT LIKE '%peach%'
ORDER BY r.rec_rating DESC;
