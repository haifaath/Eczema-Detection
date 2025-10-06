<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eczema Detector - Doctor Info</title>
<link rel="stylesheet" href="styles.css">
<link rel="stylesheet" href="style.css">
<script>
function showEditForm() {
  document.getElementById('display-info').style.display = 'none';
  document.getElementById('edit-info').style.display = 'block';
}

function hideEditForm() {
  document.getElementById('edit-info').style.display = 'none';
  document.getElementById('display-info').style.display = 'block';
}
</script>
</head>


<body>

<div class="container">
  <header>
    <h1>Eczema Detector</h1>
    <div class="user-info">
    
    </div>
  </header>

  <aside class="sidebar">
    <nav>
      <ul>
        <li><a href="adminhomepage.php">Homepage</a></li>
        <li><a href="admin_profile_page.php">My Profile</a></li>
        <li><a href="logout.php">Logout</a></li>
      </ul>
    </nav>
    <div class="support">
      <a href="#">Help & Support</a>
    </div>
  </aside>

  <main>
    <?php
    //db configuration
    $servername = "localhost";
    $username = "root";
    $password = "root"; 
    $dbname = "eczema_detection";
    $conn = new mysqli($servername, $username, $password, $dbname);
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['update'])) {
        $doctor_id = intval($_POST['doctorID']);
        $name = $_POST['doctorFirstName'] ?? null;
        $dob = $_POST['doctorDOB'] ?? null;
        $email = $_POST['doctorEmail'] ?? null;

        $stmt = $conn->prepare("UPDATE doctor SET doctorFirstName=?, doctorDOB=?, doctorEmail=? WHERE doctorID=?");
        $stmt->bind_param("sssi", $name, $dob, $email, $doctor_id);
        $stmt->execute();

        if ($stmt->affected_rows > 0) {
            echo "<script>alert('Doctor information updated successfully.');</script>";
        } else {
            echo "<script>alert('No changes were made to the doctor information.');</script>";
        }
        $stmt->close();
    }

    ?>
    <div class="doctors-list">
      <h2>Doctors' Info Homepage</h2>
      <a href="admin_add_doctor.php">
        <button type="button">Add Doctor</button>
      </a>
      <ul>
        <?php
        $sql = "SELECT doctorID, doctorFirstName FROM doctor";
        $result = $conn->query($sql);
        if ($result->num_rows > 0) {
            while($row = $result->fetch_assoc()) {
                echo "<li><a href='?doctorID=" . $row["doctorID"] . "'>" . htmlspecialchars($row["doctorFirstName"]) . "</a></li>";
            }
        } else {
            echo "<li>No doctors found</li>";
        }
        ?>
      </ul>
    </div>

    <div class="patient-info">
      <?php
      if(isset($_GET['doctorID'])) {
        $doctor_id = intval($_GET['doctorID']);
        $stmt = $conn->prepare("SELECT * FROM doctor WHERE doctorID = ?");
        $stmt->bind_param("i", $doctor_id);
        $stmt->execute();
        $result = $stmt->get_result();
        if($result->num_rows === 1) {
            $patient = $result->fetch_assoc();
            ?>
            <div id="display-info">
              <h3>Name: <?php echo htmlspecialchars($patient["doctorFirstName"]); ?></h3>
              <p>Date of Birth: <?php echo htmlspecialchars($patient["doctorDOB"]); ?></p>
              <p>Email: <a href='mailto:<?php echo htmlspecialchars($patient["doctorEmail"]); ?>'><?php echo htmlspecialchars($patient["doctorEmail"]); ?></a></p>
              <button onclick="showEditForm()">Edit Info</button>
            </div>
            <div id="edit-info" style="display:none;">
              <form method="post">
                <input type="hidden" name="doctorID" value="<?php echo $doctor_id; ?>">
                Name: <input type="text" name="doctorFirstName" value="<?php echo htmlspecialchars($patient["doctorFirstName"]); ?>"><br>
                Date of Birth: <input type="date" name="doctorDOB" value="<?php echo htmlspecialchars($patient["doctorDOB"]); ?>"><br>
                Email: <input type="email" name="doctorEmail" value="<?php echo htmlspecialchars($patient["doctorEmail"]); ?>"><br>
                <button type="submit" name="update">Update</button>
                <button type="button" onclick="hideEditForm()">Cancel</button>
              </form>
            </div>
            <?php
            $stmt->close();
        } else {
            echo "<p>Doctor not found.</p>";
        }
    } else {
        echo "<p>Please select a doctor from the list.</p>";
    }
    $conn->close();
      ?>
    </div>
  </main>
</div>

</body>
</html>
