<?php
session_start(); // Start the session

// Unsetting all the sess variables
$_SESSION = array();

// Destroy the session.
session_destroy();

// Redirect to the login page or home page after logout
header("Location: Loginpage.php");
exit;
