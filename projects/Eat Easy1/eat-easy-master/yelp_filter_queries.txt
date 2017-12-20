USE yelp_db;

DROP TABLE IF EXISTS business_AZ, uniq_business, review_AZ, tip_AZ, address_AZ;

CREATE TABLE uniq_business AS (
SELECT DISTINCT b.id
FROM business as b INNER JOIN category as c ON b.id = c.business_id 
WHERE b.state LIKE 'AZ' AND b.city LIKE 'Phoenix' AND  (c.category LIKE 'Rest%' OR c.category LIKE '%Food%'));


CREATE TABLE business_AZ AS (
SELECT b.*
FROM business as b 
INNER JOIN uniq_business as uniq
ON b.id = uniq.id
);

CREATE TABLE review_AZ AS (
SELECT b.*
FROM review as b 
INNER JOIN uniq_business as uniq
ON b.id = uniq.id
);

CREATE TABLE tip_AZ AS (
SELECT b.*
FROM tip as b 
INNER JOIN uniq_business as uniq
ON b.business_id = uniq.id
);


CREATE TABLE address_AZ AS (
SELECT b.name, b.address, b.city, b.state, b.postal_code
FROM business_AZ as b

);

select * from address_AZ;
