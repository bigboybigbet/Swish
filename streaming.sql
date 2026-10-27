/* 
SCHEMAS DESIGN
users
artists
albums
tracks
streams
subscriptions
*/
constraint:
    not null
    unique
    check
    default

data types:
    varchar
    numeric
    timestamptz 
    


CREATE TABLE users (
  id serial primary key,
  username varchar(50) not null unique,
  email    varchar(50) not null unique,
  plan     varchar(10) check(plan in ('free','premium'))
  created  timestamptz default now();
)

HW: create table, no more than 6 information each table

/*
NORMALIZATION
EX: 
track_id | title              | artist_name | artist_country
1        | Bohemian Rhapsody  | Queen       | UK
2        | We Will Rock You   | Queen       | UK
3        | Radio Ga Ga        | Queen       | UK

The 3rd collum should be artist_id and have another table 

3 rules when you create schemas:
    1NF - 1 normal form: value should be atomic, should not be grouped (if i have a collum name genre, genre value could be 'pop', 'rock', 'ballad', then with 1NF, we should create genre as a table)
    2NF - 1NF N: tat ca cac truong/cot ma khong phai PK thi phai phu thuoc vao WHOLE PK
    3NF - 2NF N: no transitive dependencies: if non PK, have to rely on PK ONLY.


2NF EX:
playlist_id | track_id | position | artist_name
3NF EX:
album_id    | title    | label    | label_address

In what situation do we denormalization?
- read-heavy system
- performance-heavy system
- analytic

In what situation do we normalization?
- write-heavy sys
- data-change frequently
- strong-consistency

INDEX: khi ma co 1 thong tin dc tim kiem nhieu, tao 1 bang an de sorting nhung thong tin trong truong thong tin do -> better performance
Ex: 
id   | Island_name   | Area     | Country_name
Ex 2: 
if the duplication is low and where,join-heavy; it is perfect for index
[5,3,6,7,1,9,11] find number 7
[1,3,5,6,7,9,11]

HW:
- Learn notion to system the knowledge
- Test incoming, topic: sql, ex: given 1 schemas, there are ques abt queries, ques abt is this optimized?, ques abt normalization, which one we should index.
- pass: 70%