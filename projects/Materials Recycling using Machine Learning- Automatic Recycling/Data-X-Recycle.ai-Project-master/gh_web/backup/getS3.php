<?php

/*
	This will pull every POS/TRN training image from S3
	and put them in dev_train/train (90%) and dev_train/validation folders (10%)
	It makes sure images are 256x256 square for training 
	Only folders >100 images are downloaded 
*/


// empty these folders before running program 
$train_dir = '/Users/geraldp/gdrive/recycle_ai/s3_train_data/train/';
$validation_dir = '/Users/geraldp/gdrive/recycle_ai/s3_train_data/valid/';
$min_number = 75;

// set time, required for S3 download
date_default_timezone_set('America/Los_Angeles'); 

// create an S3Client
include '/Users/geraldp/gdrive/recycle_ai/gh_web/vendor/autoload.php';
use Aws\S3\S3Client;
use Aws\S3\Exception\S3Exception;
$s3Client = new S3Client([
    'version'     => 'latest',
    'region'      => 'us-west-1',
    'credentials' => [
        'key'    => 'AKIAIU6EYHUUYOI5RCBA',
        'secret' => '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'
    ],
]);
   	  
// connect to the database
include '/Users/geraldp/gdrive/recycle_ai/gh_web/dashboard/ssi/db_mysqli.php';

// function to get photo_stamp from photo_id
include '/Users/geraldp/gdrive/recycle_ai/gh_web/dashboard/php/getPhotostamp.php';

// location of S3 photos 
$server = 'https://s3-us-west-1.amazonaws.com/rai-objects/';
$bucket = 'rai-objects';   

// the number of good object folders
$object_count = 0;

$sql="SELECT object_id from object";   
$rs=$conn->query($sql);
// $row_cnt = $rs->num_rows;
while($row = $rs->fetch_assoc()) {
	$object_id = $row["object_id"];	
	
	// get only the positive training photos 
	$folder = $object_id.'/POS/TRN/';

	$response = $s3Client->listObjects(array(
		'Bucket' => $bucket, 
		'Prefix' => $folder
	));
	$files = $response->getPath('Contents');
	
	// number of images in object folder, set aside 10% for validation
	$cnt_number = count($files);
	$val_number = round($cnt_number/10);
	$trn_number = $cnt_number - $val_number;

	$i = 0;

	// train on folders with > 100 images, and not /0 speedtag folder  
	if ($cnt_number > $min_number && $object_id != 0) {
	# if ($object_id == 112) {

		$object_count++;

		foreach ($files as $file) {

	   	 	// the image on S3 (ex. 106/POS/TRN/20161231101455-222856-256.jpg)
	   	 	$photo_id = $file['Key'];

	   	 	// the full S3 url for the image on S3 
			$urlname = $server.$photo_id;

	   	 	// get the photo_stamp (ex. 20161231101455-222856)
	   	 	$photo_stamp = getPhotostamp($photo_id);

	   	 	// make new object folders if they don't exist 
	   	 	if (!file_exists($train_dir.$object_id.'/')) {
			    mkdir($train_dir.$object_id.'/', 0777, true);
			}
			if (!file_exists($validation_dir.$object_id.'/')) {
			    mkdir($validation_dir.$object_id.'/', 0777, true);
			}

			$extension = end(explode(".", $photo_id));

			if ($extension == 'jpg') {

				// validation folder with 10% of images 
		   	 	if ($i <= $val_number) {
			   	 	
			   	 	// copy S3 image to this local url  
			   	 	$localurl = $validation_dir.$object_id.'/'.$object_id.'-'.$photo_stamp.'-256.jpg';
			   	 	copy($urlname, $localurl);

			   	 	// make sure the local image is 256x256
			   	 	squareImage($localurl);

			   	// train folder 
			   	} else {

			   		// copy S3 image to this local url  
			   	 	$localurl = $train_dir.$object_id.'/'.$object_id.'-'.$photo_stamp.'-256.jpg';
			   	 	copy($urlname, $localurl);

			   	 	// make sure the local image is 256x256
			   	 	squareImage($localurl);
			   	 	
				}
			}

			$i++;

		}

		echo $object_id." is ".$cnt_number." images, downloaded \n";
	}  else {
		echo $object_id." is ".$cnt_number." images, <100, not downloaded \n";
	}

}
echo "\n".$object_count." object folders downloaded";

// function to center square image to 256x256 if not already that 
function squareImage($localurl) {

	$image = imagecreatefromjpeg($localurl); 
	list($width1, $height1) = getimagesize($localurl);

	// future - make sure it is a valid image
	// if not, delete it on S3 

	if ($width1 != 256 || $height1 != 256) {

		// it's a horizontal image
		if ($width1 > $height1) {
		    $square = $height1;      // square side length is height
		    $offsetX = ($width1 - $height1) / 2;  // x offset to center square
		    $offsetY = 0;           
		}
		// it's a vertical image
		elseif ($height1 > $width1) {
		    $square = $width1;  // square side length is width
		    $offsetX = 0;
		    $offsetY = ($height1 - $width1) / 2;  // x offset to center square
		}
		// it's already a square
		else {
		    $square = $width1;
		    $offsetX = $offsetY = 0;
		}
		
		$endSize = 256;
		$tn = imagecreatetruecolor($endSize, $endSize);
		imagecopyresampled($tn, $image, 0, 0, $offsetX, $offsetY, $endSize, $endSize, $square, $square);
		imagejpeg($tn, $localurl, 100) ; 
	} 
}

		
?>