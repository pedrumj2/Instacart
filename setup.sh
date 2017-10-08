#!/bin/bash



git submodule update --init --recursive
#(cd LinuxVMSetup
#  chmod +x deep_learning_setup.sh
#  ./deep_learning_setup.sh
#)
#
#rm -rf downloads
#mkdir downloads
#(cd downloads
#  gdrive download 0Bwn99sXw-O9PRS1nbk5tVHVjMXc
#  unzip data.zip
#  rm data.zip
#  (cd mysql_csv_import
#    chmod +x script.sh
#  )
#  
#  mysql -u root -pfafdRE33 -e "Drop Schema if exists insta"
#  mysql -u root -pfafdRE33 -e "create Schema insta"
#
#  awk  -F'"' -v OFS='' '{ for (i=2; i<=NF; i+=2) gsub(",", "", $i) } 1' products.csv >products2.csv
#  mv products2.csv products.csvW
#  
#  
#  ./../mysql_csv_import/script.sh fafdRE33 insta aisles.csv &
#  ./../mysql_csv_import/script.sh fafdRE33 insta departments.csv &
#  ./../mysql_csv_import/script.sh fafdRE33 insta order_products__prior.csv &
#  ./../mysql_csv_import/script.sh fafdRE33 insta order_products__train.csv &
#  ./../mysql_csv_import/script.sh fafdRE33 insta orders.csv &
#  ./../mysql_csv_import/script.sh fafdRE33 insta products.csv &
#  ./../mysql_csv_import/script.sh fafdRE33 insta sample_submission.csv &
#  
#  wait
#    echo "finished inserting tables"
#)
#
#mysql -u root -pfafdRE33 -e"
#  update insta.orders
#  set days_since_prior_order = 0
#  where days_since_prior_order = \"\""
#echo "set days_since_prior_order to zero where null"  
#
#mysql -u root -pfafdRE33 -e" ALTER TABLE insta.order_products__prior 
#  CHANGE COLUMN order_id id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN product_id product_id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN add_to_cart_order add_to_cart_order INT NULL DEFAULT NULL ,
#  CHANGE COLUMN reordered reordered INT NULL DEFAULT NULL ,
#  ADD INDEX id (id ASC);"
#echo "changed order_products_prior table column types"
#  
#  
#echo "starting parallel process to change order_products__train table column types"
#mysql -u root -pfafdRE33 -e" ALTER TABLE insta.order_products__train 
#  CHANGE COLUMN order_id id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN product_id product_id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN add_to_cart_order add_to_cart_order INT NULL DEFAULT NULL ,
#  CHANGE COLUMN reordered reordered INT NULL DEFAULT NULL ,
#  ADD INDEX id (id ASC);" &
#
#echo "starting parallel process to change orders table column types"
#mysql -u root -pfafdRE33 -e" ALTER TABLE insta.orders 
#  CHANGE COLUMN order_id id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN user_id user_id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN order_number order_number INT NULL DEFAULT NULL ,
#  CHANGE COLUMN order_dow order_dow INT NULL DEFAULT NULL ,
#  CHANGE COLUMN order_hour_of_day order_hour_of_day INT NULL DEFAULT NULL ,
#  CHANGE COLUMN days_since_prior_order days_since_prior_order INT NULL DEFAULT NULL ,
#  ADD INDEX id (id ASC),
#  ADD INDEX I1 (eval_set ASC), 
#  ADD INDEX I2 (user_id ASC);"   &
#   
#echo "starting parallel process to change products table column types"
#mysql -u root -pfafdRE33 -e "
#  ALTER TABLE insta.products 
#  CHANGE COLUMN product_id id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN aisle_id aisle_id INT NULL DEFAULT NULL ,
#  CHANGE COLUMN department_id department_id INT NULL DEFAULT NULL ,
#  ADD INDEX id (id ASC);" &
#
#wait
#echo "parallel processed joined"
#
#
#mysql -u root -pfafdRE33 -e "
#  CREATE TABLE insta.combined 
#  SELECT product_id,  add_to_cart_order, reordered, a.user_id, a.order_number, a.order_dow, a.order_hour_of_day, a.days_since_prior_order, b.aisle_id, b.department_id, b.product_name FROM insta.order_products__prior
#  inner join insta.orders as a
#  	on order_products__prior.id = a.id
#  inner join insta.products as b
#  	on order_products__prior.product_id  = b.id
#  where a.eval_set = \"prior\";"
#echo "combined table created"
#
#
#mysql -u root -pfafdRE33 -e "
#  insert into insta.combined
#  SELECT product_id,  add_to_cart_order, reordered, a.user_id, a.order_number, a.order_dow, a.order_hour_of_day, a.days_since_prior_order, b.aisle_id, b.department_id, b.product_name FROM insta.order_products__train
#  inner join insta.orders as a
#  	on order_products__train.id = a.id
#  inner join insta.products as b
#  	on order_products__train.product_id  = b.id
#  where a.eval_set = \"train\";"
#echo "train values inserted into combined table"
#
#mysql -u root -pfafdRE33 -e "
#  ALTER TABLE insta.combined 
#  ADD INDEX I1 (user_id ASC, order_number ASC, add_to_cart_order ASC), 
#  ADD INDEX I2 (department_id ASC);"
#echo "index added to combined"
# 
#
#
##department 22 does not exist
#mysql -u root -pfafdRE33 -e "
#  update insta.combined
#    set department_id = 22
#    where department_id is null"
#    
#echo "updated null department ids"