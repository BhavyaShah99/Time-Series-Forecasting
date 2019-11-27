DROP SCHEMA IF EXISTS schema cascade;
CREATE SCHEMA schema;
SET search_path TO schema, public;

CREATE TABLE Guests (
  guest_id integer PRIMARY KEY NOT NULL,
  surname varchar(25) NOT NULL,
  firstname varchar(15) NOT NULL,
  dob date NOT NULL,
  address varchar(50) NOT NULL
);

CREATE TABLE Hosts (
  email varchar(30) PRIMARY KEY NOT NULL
);

CREATE TABLE Property (
  property_id integer PRIMARY KEY NOT NULL,
  host varchar(30) NOT NULL REFERENCES Hosts(email),
  bedrooms integer NOT NULL,
  bathrooms integer NOT NULL,
  capacity varchar(10) NOT NULL,
  address varchar(50) NOT NULL,
  price integer NOT NULL,
  stable_weekly_price boolean NOT NULL DEFAULT false,
  CONSTRAINT min_capacity CHECK (CAST(RIGHT(capacity, 2) AS int) >= bedrooms) --how to get the number out of the capacity
);

CREATE TABLE PropertyType (
  property_id integer PRIMARY KEY NOT NULL REFERENCES Property,
  property_type varchar(10) NOT NULL,
  walkability score,
  closest_transit varchar(10),
  water_type varchar(10),
  lifegaurd_offered boolean DEFAULT false,
  CONSTRAINT property_type CHECK(property_type == 'city' OR property_type == 'water' OR property_type == 'city-water')
  CONSTRAINT transit_type CHECK (closest_transit == 'bus' OR closest_transit == 'LRT' OR closest_transit == 'subway' OR closest_transit == 'none'),
  CONSTRAINT water_type CHECK (water_type == 'beach' OR water_type == 'lake' OR water_type == 'pool')
);

CREATE TABLE Luxuries (
  property_id integer NOT NULL,
  luxury varchar(20) NOT NULL
  PRIMARY KEY (property_id, luxury),
  CONSTRAINT luxuries CHECK (luxury == 'hot tub' OR luxury == 'sauna' OR luxury == 'laundry service' OR luxury == 'daily cleaning' OR 
                             luxury == 'daily breakfast delivery' OR luxury == 'concierge service')
);

CREATE TABLE AvailableProperties (
  property_id integer PRIMARY KEY NOT NULL REFERENCES Property,
  datetime date
);

CREATE TABLE Rental (
  rental_id integer PRIMARY KEY NOT NULL,
  renter integer REFERENCES Guests(guest_id),
  property_id integer REFERENCES Property
  num_of_people integer NOT NULL,
  check_in_date date,
  check_out_date date,
  renter_creadit_card bigint NOT NULL,
  CONSTRAINT within_capating CHECK(num_of_people <= property_id(capacity)),
  CONSTRAINT renter_age CHECK(renter(age) >= 18), --subtract checkin date from renter dob to check if below 18
  CONSTRAINT dates_order CHECK(check_in_date <= check_out_date)
);

CREATE TABLE RentalGuestsInfo (
  rental_id integer NOT NULL REFERENCES Rental,
  guest_id integer NOT NULL REFERENCES Guests,
  PRIMARY KEY (rental_id, guest_id)
);

CREATE TABLE PropertyRating (
  guest_id integer NOT NULL REFERENCES Guests,
  rental_id integer PRIMARY KEY NOT NULL REFERENCES Rental,
  comment varchar(50) NOT NULL,
  stars rating NOT NULL
);

CREATE TABLE HostRating (
  renter integer NOT NULL REFERENCES Rental,
  rental_id integer PRIMARY KEY NOT NULL REFERENCES Rental,
  stars rating NOT NULL
);

CREATE DOMAIN score AS smallint 
   DEFAULT NULL
   CHECK (VALUE >= 0 AND VALUE <= 100);

CREATE DOMAIN rating AS smallint 
   DEFAULT NULL
   CHECK (VALUE >= 0 AND VALUE <= 5);