#!/bin/bash




cd ..

mysql -u root -pfafdRE33 -e "
    drop schema if exists insta2;"
    
mysql -u root -pfafdRE33 -e "
    CREATE TABLE insta2.combined
    SELECT *
    FROM insta.combined
    where user_id =1;"    