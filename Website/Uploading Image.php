<!DOCTYPE html>
<html>
<head>
    <title>Homepage</title>
    <link rel="stylesheet" href="style.css">
    <script>
    function validateForm() {
        var isValid = true;
        var firstname = document.getElementById('patientfirstname').value;
        var lastname = document.getElementById('patientlastname').value;
        var letterRegex = /^[A-Za-z]+$/;

        document.getElementById('firstnameError').innerText = "";
        document.getElementById('lastnameError').innerText = "";

        if (firstname === "") {
            document.getElementById('firstnameError').innerText = 'First name is required.';
            isValid = false;
        } else if (!letterRegex.test(firstname)) {
            document.getElementById('firstnameError').innerText = 'First name must contain only letters.';
            isValid = false;
        }

        if (lastname === "") {
            document.getElementById('lastnameError').innerText = 'Last name is required.';
            isValid = false;
        } else if (!letterRegex.test(lastname)) {
            document.getElementById('lastnameError').innerText = 'Last name must contain only letters.';
            isValid = false;
        }
        return isValid;
    }
    function showSpinner() {
            document.getElementById('spinner').style.display = 'block';
            document.getElementById('overlay').style.display = 'block';
        }
    </script>
        <style>

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            display: none;
            border: 16px solid #f3f3f3;
            border-top: 16px solid #3498db;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            animation: spin 2s linear infinite;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 1001;
        }

        .overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5); /* semi-transparent background */
            backdrop-filter: blur(5px); /* optional blur effect */
            z-index: 1000;
        }
    </style>
</head>
<body class="container">
    <?php
    session_start();
    include 'db_connection.php'; 
    if (!isset($_SESSION["loggedin"]) || $_SESSION["loggedin"] !== true) {
        header("Location: Loginpage.php");
        exit;
    }
    if (isset($_SESSION["role"]) && $_SESSION["role"]=='doctor') {
        $email = $_SESSION['doctor_email'];
        
        // Fetch doctor's details using the email
    $stmt = $conn->prepare("SELECT doctorID, doctorFirstName FROM doctor WHERE doctorEmail = ?");
    $stmt->bind_param("s", $email);
    $stmt->execute();
    $result = $stmt->get_result();
    $doctorDetails = $result->fetch_assoc();
    $doctorID = $doctorDetails['doctorID'];
    $doctorFirstName = $doctorDetails['doctorFirstName'];
    $stmt->close();
    $conn->close();
    ?>
    <div class="admin-homepage-sidebar">
        <h2 class="sidebar-title">Eczema Detector</h2>
        <nav class="sidebar-nav">
            <a href="doctorhomepage.php"><i class="fas fa-home"></i> Homepage</a>
            <a href="dr_profile_page.php"><i class="fas fa-user"></i> My Profile</a>
            <a href="Uploading Image.php" class="active"><i class="fas fa-sign-out-alt"></i> Upload Image</a>
            <a href="logout.php"><i class="fas fa-sign-out-alt"></i> Logout</a>
        </nav>
    </div>
    <div class="right-panel-admin-homepage">
        <div class="right-panel-admin-homepage-header">
            <div class="header-title">Upload Image</div>
            <div class="header-user">
                <span class="user-greeting">Hello, Dr. <?php echo htmlspecialchars($doctorFirstName); ?></span>
            </div>
        </div>
        <div class="admin-homepage-lower-part">
        <form action="http://127.0.0.1:5541/upload" method="post" enctype="multipart/form-data" class="center-align-form" onsubmit="return validateForm()">
            <div class="input-group-uploadimage">
                    <p class="edit-info-p">Patient first name</p>
                      <input type="text" id="patientfirstname" name="patientfirstname" placeholder="First name" required>
                      <span id="firstnameError" class="error"></span>
            </div>
            <div class="input-group-uploadimage">
                    <p class="edit-info-p">Patient last name</p>
                      <input type="text"id="patientlastname" name="patientlastname" placeholder="Last name" required>
                      <span id="lastnameError" class="error"></span>
            </div>
            <div class="input-group-uploadimage">
                    <p class="edit-info-p">Any comments?</p>
                      <input type="text" name="doctorcomments"id="doctorcomment" placeholder="Comments.."ß>
            </div>
            <input type="hidden" name="user_id" value="<?php echo $doctorID?>">
            <div class="center-align-div-submit-btn">
            <input type="file" name="image" required>
            <input type="hidden" name="user_id" value="<?php echo $doctorID?>">
            <input type="submit" value="Upload Image" name="submit" class="btn-login">
        </form>
        <?php 
        if (isset($_GET['status'])) {
        if ($_GET['status']=='success') {
            echo '<p class="success"> Image Uploaded successfully. View your results <a href="http://localhost:8888/doctor_prev_recs.php" >here</a></p>';
        }
    else {
        echo '<p class="failed"> Something went wrong. Please try again</p>';
    } }
        
           if (isset($_GET['result']) && isset($_GET['filename'])) {
        $result = htmlspecialchars($_GET['result']);
        $filename = htmlspecialchars($_GET['filename']); ?>

    <p>Result: <?php echo isset($_GET['result']) ? $_GET['result'] : 'No result received.'; ?></p>
   <?php echo "<img src='my_flask_app/uploads/" . $filename . "' alt='Processed Image'/>"; ?>
    </div>
        </div>
    </div>
        <?php
    }}
    //$_SESSION['loggedin'] = true;
    //$_SESSION['role'] = 'patient';
    else if (isset($_SESSION["role"]) && $_SESSION["role"]=='patient') {
        $email = $_SESSION['patient_email'];
        // Fetch patient's details using the email
        $stmt = $conn->prepare("SELECT patientID, patientFirstName FROM patient WHERE patientEmail = ?");
        $stmt->bind_param("s", $email);
        $stmt->execute();
        $result = $stmt->get_result();
        $patientDetails = $result->fetch_assoc();
        $patientID = $patientDetails['patientID'];
        $patientFirstName = $patientDetails['patientFirstName'];
        $stmt->close();
        $conn->close();
    ?>
    <div class="admin-homepage-sidebar">
        <h2 class="sidebar-title">Eczema Detector</h2>
        <nav class="sidebar-nav">
            <a href="#" class="active"><i class="fas fa-home"></i> Homepage</a>
            <a href="http://localhost:8888/patient_view_prev_results.php"><i class="fas fa-user"></i> All Results</a>
            <a href="patient_profile_page.php"><i class="fas fa-user"></i> My Profile</a>
            <a href="logout.php"><i class="fas fa-sign-out-alt"></i> Logout</a>
        </nav>
    </div>
    <div class="right-panel-admin-homepage">
        <div class="right-panel-admin-homepage-header">
            <div class="header-title">Homepage</div>
            <div class="header-user">
                <span class="user-greeting">Hello, <?php echo htmlspecialchars($patientFirstName); ?></span>
            </div>
        </div>
        <div class="admin-homepage-lower-part">
            <h1 class="blue">Upload an image to get a diagnosis</h1>
            <br>
            <img src="arrow-bottom-svgrepo-com (1).svg" style="width:5em; height:6em"></img>
            <br>
        <div>
        <form action="http://127.0.0.1:5541/upload" method="post" enctype="multipart/form-data" class="center-align-form" onsubmit="showSpinner()">
            <input type="file" name="image" required>
            <br>
            <input type="submit" value="Upload Image" name="submit" class="btn-upload">
            <input type="hidden" name="user_id" value="<?php echo $patientID?>">
        </form>
        <div id="overlay" class="overlay"></div>
    <div id="spinner" class="spinner"></div>
        <?php
        if (isset($_GET['status'])) {
            if ($_GET['status']=='success') {
                echo '<br><br><p class="success"> Image uploaded successfully! View your results <a href="http://localhost:8888/patient_view_prev_results.php" >here</a></p>';
            }
            else {
                echo '<p class="failed"> Something went wrong. Please try again</p>';
                } }
         ?>
    </div>
        </div>
    </div>
   <?php }?>
    
</body>
</html>
