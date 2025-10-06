<?php
session_start();
require_once "db_connection.php"; // Database connection file

// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['new_password'])) {
    $new_password = mysqli_real_escape_string($conn, $_POST['new_password']);
    // $email = $_SESSION['reset_email'];

    // Hash the new password
    $hashed_password = password_hash($new_password, PASSWORD_DEFAULT);

    // Update the user's password
    $find_admin =  "SELECT * FROM admin WHERE adminEmail = '$_SESSION[email_to_reset]'";
    $find_patient = "SELECT * FROM patient WHERE patientEmail = '$_SESSION[email_to_reset]'";
    $find_doctor = "SELECT * FROM doctor WHERE doctorEmail = '$_SESSION[email_to_reset]'";
    $admin_result= mysqli_query($conn, $find_admin);
    $patient_result= mysqli_query($conn, $find_patient);
    $admin_result= mysqli_query($conn, $find_admin);
    $doctor_result= mysqli_query($conn, $find_doctor);
    if (mysqli_num_rows($admin_result) ==1) {
        $query = "UPDATE admin SET adminPassword = ? WHERE adminEmail = ?";
    }
    if (mysqli_num_rows($patient_result) ==1) {
        $query = "UPDATE patient SET patientPassword = ? WHERE patientEmail = ?";
    }
    if (mysqli_num_rows($doctor_result) ==1) {
        $query = "UPDATE doctor SET doctorPassword = ? WHERE doctorEmail = ?";
    }
    //DO OTHER USERS TMWWWW
$query = "UPDATE admin SET adminPassword = ? WHERE adminEmail = ?"; }

// Prepare the statement to prevent SQL injection
if ($stmt = mysqli_prepare($conn, $query)) {
    // Bind variables to the prepared statement as parameters
    mysqli_stmt_bind_param($stmt, "ss", $new_password, $_SESSION['email_to_reset']);
    
    // Attempt to execute the prepared statement
    if (mysqli_stmt_execute($stmt)) {
        // Success, password updated
        echo "Password updated successfully.";
        
        // Unset the session variables used for password reset
        unset($_SESSION['email_to_reset']);
        unset($_SESSION['reset_code']);
        
        // Redirect the user to the login page, or a confirmation page
        header("Location: Loginpage.php?pass=update");
        exit();
    } else {
        echo "Oops! Something went wrong. Please try again later.";
    }

    // Close statement
    mysqli_stmt_close($stmt);
} else {
    // Could not prepare the query statement
    echo "ERROR: Could not prepare query: $sql. " . mysqli_error($link);
}

// Close database connection
mysqli_close($link);