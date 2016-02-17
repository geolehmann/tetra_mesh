#pragma once
#include "Math.h"

struct Camera 
{
	float2 resolution;
	float4 position;
	float4 view;
	float4 up;
	float2 fov;
	float apertureRadius;
	float focalDistance;
};

// class for interactive camera object, updated on the CPU for each frame and copied into Camera struct
class InteractiveCamera
{
private:

	float4 centerPosition;
	float4 viewDirection;
	float yaw;
	float pitch;
	float radius;
	float apertureRadius;
	float focalDistance;

	void fixYaw();
	void fixPitch();
	void fixRadius();
	void fixApertureRadius();
	void fixFocalDistance();

public:

	float4 getPos(){ return centerPosition; }
	float4 getDir(){ return viewDirection; }

	InteractiveCamera();
	virtual ~InteractiveCamera();
	void changeYaw(float m);
	void changePitch(float m);
	void changeRadius(float m);
	void changeAltitude(float m);
	void changeFocalDistance(float m);
	void strafe(float m);
	void goForward(float m);
	void rotateRight(float m);
	void changeApertureDiameter(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCamera(Camera* renderCamera);

	float2 resolution;
	float2 fov;
};

InteractiveCamera::InteractiveCamera()
{
	centerPosition = make_float4(0.0f, 0.0f, 0.0f, 0);
	yaw = 0.0;
	pitch = 0.1;
	radius = 4;
	apertureRadius = 0.0; // 0.04
	focalDistance = 4.0f;

	resolution = make_float2(512, 512);  // width, height
	fov = make_float2(40, 40);
}

InteractiveCamera::~InteractiveCamera() {}

void InteractiveCamera::changeYaw(float m){
	yaw += m;
	fixYaw();
}

void InteractiveCamera::changePitch(float m){
	pitch += m;
	fixPitch();
}

void InteractiveCamera::changeRadius(float m){
	radius += radius * m; // Change proportional to current radius. Assuming radius isn't allowed to go to zero.
	fixRadius();
}

void InteractiveCamera::changeAltitude(float m){
	centerPosition.y += m;
	//fixCenterPosition();
}

void InteractiveCamera::goForward(float m){
	centerPosition += viewDirection * m;
}

void InteractiveCamera::strafe(float m){
	float4 strafeAxis = CrossCPU(viewDirection, make_float4(0, 1, 0, 0));
	strafeAxis = normalizeCPU(strafeAxis);
	centerPosition += strafeAxis * m;
}

void InteractiveCamera::rotateRight(float m){
	float yaw2 = yaw;
	yaw2 += m;
	float pitch2 = pitch;
	float xDirection = sin(yaw2) * cos(pitch2);
	float yDirection = sin(pitch2);
	float zDirection = cos(yaw2) * cos(pitch2);
	float4 directionToCamera = make_float4(xDirection, yDirection, zDirection, 0);
	viewDirection = directionToCamera * (-1.0);
}

void InteractiveCamera::changeApertureDiameter(float m){
	apertureRadius += (apertureRadius + 0.01) * m; // Change proportional to current apertureRadius.
	fixApertureRadius();
}


void InteractiveCamera::changeFocalDistance(float m){
	focalDistance += m;
	fixFocalDistance();
}


void InteractiveCamera::setResolution(float x, float y){
	resolution = make_float2(x, y);
	setFOVX(fov.x);
}

float radiansToDegrees(float radians) {
	float degrees = radians * 180.0 / PI;
	return degrees;
}

float degreesToRadians(float degrees) {
	float radians = degrees / 180.0 * PI;
	return radians;
}

void InteractiveCamera::setFOVX(float fovx){
	fov.x = fovx;
	fov.y = radiansToDegrees(atan(tan(degreesToRadians(fovx) * 0.5) * (resolution.y / resolution.x)) * 2.0);
	// resolution float division
}

void InteractiveCamera::buildRenderCamera(Camera* renderCamera){
	float xDirection = sin(yaw) * cos(pitch);
	float yDirection = sin(pitch);
	float zDirection = cos(yaw) * cos(pitch);
	float4 directionToCamera = make_float4(xDirection, yDirection, zDirection, 0);
	viewDirection = directionToCamera * (-1.0);
	float4 eyePosition = centerPosition + directionToCamera * radius;
	//float4 eyePosition = centerPosition; // rotate camera from stationary viewpoint


	renderCamera->position = eyePosition;
	renderCamera->view = viewDirection;
	renderCamera->up = make_float4(0, 1, 0, 0);
	renderCamera->resolution = make_float2(resolution.x, resolution.y);
	renderCamera->fov = make_float2(fov.x, fov.y);
	renderCamera->apertureRadius = apertureRadius;
	renderCamera->focalDistance = focalDistance;
}

float mod(float x, float y) { // Does this account for -y ???
	return x - y * floorf(x / y);
}

void InteractiveCamera::fixYaw() {
	yaw = mod(yaw, 2 * PI); // Normalize the yaw.
}

float clamp2(float n, float low, float high) {
	n = fminf(n, high);
	n = fmaxf(n, low);
	return n;
}

void InteractiveCamera::fixPitch() {
	float padding = 0.05;
	pitch = clamp2(pitch, -PI_OVER_TWO + padding, PI_OVER_TWO - padding); // Limit the pitch.
}

void InteractiveCamera::fixRadius() {
	float minRadius = 0.2;
	float maxRadius = 100.0;
	radius = clamp2(radius, minRadius, maxRadius);
}

void InteractiveCamera::fixApertureRadius() {
	float minApertureRadius = 0.0;
	float maxApertureRadius = 25.0;
	apertureRadius = clamp2(apertureRadius, minApertureRadius, maxApertureRadius);
}

void InteractiveCamera::fixFocalDistance() {
	float minFocalDist = 0.2;
	float maxFocalDist = 100.0;
	focalDistance = clamp2(focalDistance, minFocalDist, maxFocalDist);
}
