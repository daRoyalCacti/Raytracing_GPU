//Axis-align bounding box class
//are parallelepipeds that are aligned to the axes that bound objects in the scene
//used to speed of light ray intersections

#pragma once

struct aabb {
	point3 minimum, maximum;	//the points that define the bounding box
					// in 1D    ->         |        |
					//        orign       min      max
					//       of ray     

	__host__ __device__ aabb() {}
	__host__ __device__ aabb(const point3& a, const point3& b) : minimum(a), maximum(b) {}

	__host__ __device__ inline point3 min() const {return minimum;}
	__host__ __device__ inline point3 max() const {return maximum;}

	__device__ inline bool hit(const ray& r, float t_min, double t_max) const {
		//Andrew Kensler (from Pixar) intersection method
		for (int i = 0; i < 3; i++) {
			const auto invD = 1.0f / r.direction()[i];	// 1/x or 1/y or 1/z for the incomming ray
			auto t0 = (min()[i] - r.origin()[i]) * invD;	//the time it takes the ray to hit the 'min' side of bounding box
									// s=d/t => t = d/s
									// d is the distance from origin to bounding box in a single direction
									// - so d = (min.x - origin.x)   (see picture above)
									// the ray is defined as r = r0 + dir*t
									// this can be taken as r = r0 + speed * time assumine the speed is 1 
									// (speed is taken to be instant so any normalisation on the dir is irrelevent here)
									// therefore, t = (min.x - orgin.x) / dir.x
										
			auto t1 = (max()[i] - r.origin()[i]) * invD;	//same as for t0 but for the 'max' side of the bounding box
			
			float temp;
			if (invD < 0.0f) {  //taking it as the ray hits 'min' first then 'max', if the ray is travelling towards the origin
				temp = t0;	//the ray will hit max first then min
				t0 = t1;	//so simplying swapping t0 and t1
				t1 = temp;	//this really only saves 1 if statement and makes the code nearter

			}
																																						//shrinking the interval that a ray can collide
			//in order for the ray to hit the box
			// - if the ray collides with min after the previous smallest collision, this is the new smallest collision
			// - similar for max
			t_min = t0 > t_min ? t0 : t_min;
			t_max = t1 < t_max ? t1 : t_max;

			if (t_max <= t_min)		//tmax and tmin get changed for each component (i.e. first for x, then for y, then for z)
				return false;		//the only way for this to return true is if the time taken for a ray to collided with 
							// - max in say the y direction is smaller than the time taken for the ray to collide with min in the x direction
							// - min in say the y direction is bigger than the time taken for the ray to collide with max in the x direction
							//in both cases there is no time for which the ray is inside the box
							// - the ray collides with max in 1 dimension then min in the other dimesion
							//   -- once it collides with max, it is outside the box
							//   -- i.e. ther ray is outside the box in 1D, inside (by colliding with min) in 1D, outside (by colliding with max), 
							//      then inside in another dimension by colliding with min
							// - still the ray collides with max in one dimension then the other
							//   -- colliding with min in the y direction after colliding with max in the x direction means
							//      the ray collided with max in the x direction first then min in the y direction
							//if by the end t_max is less than t_min, the ray is inside the box for t in [t_min, t_max]
							// - the time for with the ray is after min but before max in all of x,y,z
							//
							//consider a 2D box and 3 possible rays (time increasing moving right -- collide with min first then max)
							// - requires tab size of 8
							//		      min.x			      max.x
							//
							//		    	|			      /	| tmax.x
							//		tmin.x /|			     /	|
							//		      /	|			    /	|
							//	             /	|			   /	|
							//	            /	|			  /	|
							//	           /	|			 /	|
							//	          /	|			/	|		 /
							//	         /	|		       /	|		/
							//	        /	|		      /		|       tmax.y /
							//-----------------------------------------------------------------------     max.y
							//    tmax.y  /		|	   tmax.y   /		|            /
							//           /		|		   /		|           /
							//          /		|		  /		|          /
							//         /		|		 /		|         /
							//        /		|		/		|        /
							//       /		|	       /		|       /
							//      /		|	      /			|      /
							//     /		|	     /			|     /
							//    /	tmin.y		|   tmin.y  /			|    /
							//-----------------------------------------------------------------------      min.y
							//  /			|	  /			|  / tmin.y
							// /			|	 /		        | /
							//			|	/		        |/
							//			|      /		      	|
							//			|     /		      tmax.x   /|
							//			|    /			      /	|
							//			|   /			     / 	|
							//			|  /			    /	|
							//			| /	tmin.x		   / 	|
							//
							//left ray does not collide becasue collides with max in the y direction before min in the x direction
							// - tmax.y < tmin.x
							//right ray does not collide because collides with min in the y direction after max in the x direction
							// - tmin.y > tmax.x
							//middle ray is fine
			}				
		return true;
	}

};

inline aabb surrounding_box(const aabb box0, const aabb box1) {		//creates a larger bounding box around 2 smaller bounding boxes
	const point3 small(	fminf(box0.min().x(),  box1.min().x()),
				fminf(box0.min().y(),  box1.min().y()),
				fminf(box0.min().z(),  box1.min().z()) );

	const point3 big(	fmaxf(box0.max().x(),  box1.max().x()),
				fmaxf(box0.max().y(),  box1.max().y()),
				fmaxf(box0.max().z(),  box1.max().z()) );

	return aabb(small, big);

}
