CREATE TABLE asset_identifiers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    market_code VARCHAR(10) NOT NULL,
    asset_code VARCHAR(20) NOT NULL,
    full_code VARCHAR(30) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    asset_name VARCHAR(100) NOT NULL,
    UNIQUE (full_code)
);