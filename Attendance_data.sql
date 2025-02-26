CREATE DATABASE IF NOT EXISTS attendance;

USE attendance;

CREATE TABLE IF NOT EXISTS attendance (
    ID INT,
    Name VARCHAR(255),
    Date DATE,
    Time TIME,
    PRIMARY KEY (ID, Date)  -- Ensures one entry per user per day
);

select*from attendance;


