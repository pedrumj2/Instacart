#!/bin/bash




mysql -u root -pfafdRE33 -e "
    drop schema if exists insta2;"
mysql -u root -pfafdRE33 -e "
    create schema insta2;"
    
mysql -u root -pfafdRE33 -e "
    CREATE TABLE insta2.combined
    SELECT *
    FROM insta.combined
    where user_id < ${1};"    
    
mysql -u root -pfafdRE33 -e "    
  CREATE TABLE insta2.products (
    id INT NOT NULL AUTO_INCREMENT,
    value INT NULL,
    PRIMARY KEY (id),
    INDEX I1 (value ASC));"

mysql -u root -pfafdRE33 -e "   
  ALTER TABLE insta2.combined 
  ADD INDEX I1 (product_id ASC), 
  ADD INDEX I2 (user_id ASC, order_number ASC);
       
  insert into insta2.products(value)
  select distinct(product_id) from insta2.combined;
  
  update insta2.combined
  inner join insta2.products
	 on value = product_id
  set product_id = id;"     



  
