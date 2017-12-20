USE yelp_db;

DROP TABLE IF EXISTS business_AZ, uniq_business, tip_AZ, address_AZ, review_temp;

CREATE TABLE uniq_business AS (
SELECT DISTINCT b.id
FROM business as b INNER JOIN category as c ON b.id = c.business_id 
WHERE b.state LIKE 'AZ' AND b.city LIKE 'Phoenix' AND  (c.category LIKE 'Rest%' OR c.category LIKE '%Food%'));


CREATE TABLE business_AZ AS (
SELECT b.*
FROM business as b, uniq_business as uniq
WHERE b.id = uniq.id
);

-- CREATE TABLE review_AZ AS (
-- SELECT r.*
-- FROM review as r
-- INNER JOIN uniq_business as uniq
-- ON r.business_id = uniq.id
-- );


CREATE TABLE tip_AZ AS (
SELECT b.*
FROM tip as b 
INNER JOIN uniq_business as uniq
ON b.business_id = uniq.id
);


CREATE TABLE address_AZ AS (
SELECT b.id, b.name, b.address, b.city, b.state, b.postal_code
FROM business_AZ as b

);

-- select * from address_AZ;
-- select * from review_AZ;

CREATE TABLE review_temp AS (
SELECT r.business_id, r.text, r.stars, AVG(r.stars) as avg
FROM review_AZ AS r
GROUP BY r.business_id, r.stars, r.text
ORDER BY r.business_id DESC
);

CREATE TABLE tip_temp AS (
SELECT t.business_id, t.text, r.avg
FROM tip_AZ AS t, review_temp as r
WHERE t.business_id = r.business_id
LIMIT 50000
);

-- select COUNT(*) from review_AZ;
-- select COUNT(DISTINCT business_id) from review_AZ;
-- select * from review_AZ ORDER BY business_id;
-- select COUNT(*) from business;


-- select b.id, b.review_count 
-- from business as b, uniq_business as u
-- where b.id = u.id; 

-- select COUNT(*) from review where business_id LIKE '-Bdw-5H5C4AYSMGnAvmnzw';
select COUNT(*) from tip_temp; 
