#include "State.hpp"
#include <thread>
#include <vector>
#include <iostream>
#include <StateRender.cuh>
#include <CArray.cuh>
#include "Timer.hpp"

#include "CSDF.cuh"


State::State() {
	render = new StateRender();
}

State State::state = State();

bool State::IsKeyDown(char keycode) {
    return keysPressed.test(keycode);
}

void State::Create() 
{
    std::cout << "HALLO3 " << std::endl;

    Timer t1("allocating");
    
    render->framebuffer = Framebuffer();
    render->framebuffer.Allocate(dispWIDTH, dispHEIGHT);

    render->distBuffer = CArray();
    render->distBuffer.Allocate((dispWIDTH/2) * (dispHEIGHT/2) * 2);

    render->cArray = CArray();
    render->cArray.Allocate(BYTESIZE);

    render->bitsArray = new uint32_t[BYTESIZE / 4];

    render->csdf = CSDF();
    render->csdf.Allocate();

    t1.s();

    Timer t2("BUILDING FINE ARRAY");
    render->cArray.fill();
    render->cArray.readback(render->bitsArray);
    t2.s();

    Timer t3("BUILDING CSDF");
    render->csdf.Generate(render->cArray);
    t3.s();
}




// #include "State.hpp"
// #include "util.hpp"
// #include "Character.hpp"
// #include "hitInfo.hpp"
// #include "Framebuffer.cuh"

// #include <thread>
// #include <vector>
// #include <bitset>
// #include <intrin.h>

// const vec3 sunDir = normalize(vec3(10.f, 5.f, -4.f));
// void State::drawPixel(int ix, int iy)
// {
//     float x = (float)ix / (float)dispWIDTH;
//     float y = (float)iy / (float)dispHEIGHT;

//     vec3 color = computeColor(x, y);
//     char r = (char)(color.x * 255.0f);
//     char g = (char)(color.y * 255.0f);
//     char b = (char)(color.z * 255.0f);

//     bmp[(ix + iy * dispWIDTH) * 4 + 2] = r;
//     bmp[(ix + iy * dispWIDTH) * 4 + 1] = g;
//     bmp[(ix + iy * dispWIDTH) * 4 + 0] = b;
// }
// void State::drawRow(int row)
// {
//     for(int i = 0; i < dispWIDTH; i++)
//         drawPixel(i, row);
// }

// void State::draw()
// {
// 	  //auto timeBefore = std::chrono::high_resolution_clock::now();

//     std::vector<std::thread> threads;
//     for(int i = 0; i < dispHEIGHT; i++) 
//         threads.push_back(std::thread(&State::drawRow, this, i));

//     // auto timeAfter = std::chrono::high_resolution_clock::now();
//     // std::chrono::duration<double, std::milli> timeTaken = timeAfter - timeBefore;
//     // cout << "adding time "<< timeTaken.count() << endl;
//     for(auto &t : threads)
//     { 
//         if(t.joinable())
//             t.join();
//     }    
// }

// float noise3D(vec3 p)
// {
// 	return glm::fract(sin(dot(p ,vec3(12.9898,78.233,128.852))) * 43758.5453)*2.0f-1.0f;
// }

// float simplex3D(vec3 p)
// {
	
// 	float f3 = 1.0/3.0;
// 	float s = (p.x+p.y+p.z)*f3;
// 	int i = int(floor(p.x+s));
// 	int j = int(floor(p.y+s));
// 	int k = int(floor(p.z+s));
	
// 	float g3 = 1.0/6.0;
// 	float t = float((i+j+k))*g3;
// 	float x0 = float(i)-t;
// 	float y0 = float(j)-t;
// 	float z0 = float(k)-t;
// 	x0 = p.x-x0;
// 	y0 = p.y-y0;
// 	z0 = p.z-z0;
	
// 	int i1,j1,k1;
// 	int i2,j2,k2;
	
// 	if(x0>=y0)
// 	{
// 		if(y0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; } // X Y Z order
// 		else if(x0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; } // X Z Y order
// 		else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }  // Z X Z order
// 	}
// 	else 
// 	{ 
// 		if(y0<z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; } // Z Y X order
// 		else if(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; } // Y Z X order
// 		else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; } // Y X Z order
// 	}
	
// 	float x1 = x0 - float(i1) + g3; 
// 	float y1 = y0 - float(j1) + g3;
// 	float z1 = z0 - float(k1) + g3;
// 	float x2 = x0 - float(i2) + 2.0*g3; 
// 	float y2 = y0 - float(j2) + 2.0*g3;
// 	float z2 = z0 - float(k2) + 2.0*g3;
// 	float x3 = x0 - 1.0 + 3.0*g3; 
// 	float y3 = y0 - 1.0 + 3.0*g3;
// 	float z3 = z0 - 1.0 + 3.0*g3;	

// 	vec3 ijk0 = vec3(i,j,k);
// 	vec3 ijk1 = vec3(i+i1,j+j1,k+k1);	
// 	vec3 ijk2 = vec3(i+i2,j+j2,k+k2);
// 	vec3 ijk3 = vec3(i+1,j+1,k+1);	

// 	vec3 gr0 = normalize(vec3(noise3D(ijk0),noise3D(ijk0*2.01f),noise3D(ijk0*2.02f)));
// 	vec3 gr1 = normalize(vec3(noise3D(ijk1),noise3D(ijk1*2.01f),noise3D(ijk1*2.02f)));
// 	vec3 gr2 = normalize(vec3(noise3D(ijk2),noise3D(ijk2*2.01f),noise3D(ijk2*2.02f)));
// 	vec3 gr3 = normalize(vec3(noise3D(ijk3),noise3D(ijk3*2.01f),noise3D(ijk3*2.02f)));

// 	float n0 = 0.0;
// 	float n1 = 0.0;
// 	float n2 = 0.0;
// 	float n3 = 0.0;

// 	float t0 = 0.5 - x0*x0 - y0*y0 - z0*z0;
// 	if(t0>=0.0)
// 	{
// 		t0*=t0;
// 		n0 = t0 * t0 * dot(gr0, vec3(x0, y0, z0));
// 	}
// 	float t1 = 0.5 - x1*x1 - y1*y1 - z1*z1;
// 	if(t1>=0.0)
// 	{
// 		t1*=t1;
// 		n1 = t1 * t1 * dot(gr1, vec3(x1, y1, z1));
// 	}
// 	float t2 = 0.5 - x2*x2 - y2*y2 - z2*z2;
// 	if(t2>=0.0)
// 	{
// 		t2 *= t2;
// 		n2 = t2 * t2 * dot(gr2, vec3(x2, y2, z2));
// 	}
// 	float t3 = 0.5 - x3*x3 - y3*y3 - z3*z3;
// 	if(t3>=0.0)
// 	{
// 		t3 *= t3;
// 		n3 = t3 * t3 * dot(gr3, vec3(x3, y3, z3));
// 	}
// 	return 96.0*(n0+n1+n2+n3);
// }

// float State::Evaluate(glm::vec3 pos)
// {
//     return simplex3D(pos * 0.005f);
// }

// void State::fillSliceOfBits(int k)
// {
// 	for(int j = 0; j < 256 * 256; j++)
// 	{
// 			vec3 pos = vec3(j & 255, j >> 8, k);
			
// 			bool solid = Evaluate(pos) > 0.7;
// 			//bool solid = glm::distance(pos, vec3(64)) < 16.0f;
// 			bits.set(j | (k << 16), solid);
// 	}
// 	// __m256i row;
// 	// for(int j = 0; j < 256; j ++)
// 	// {
// 	// 	_mm256_xor_si256(row, row);		
// 	// 	for(int i = 0; i < 256; i++)
// 	// 	{
// 	// 		vec3 pos = vec3(j & 255, j >> 8, k);
// 	// 		unsigned int solid = simplex3D(0.05f * pos) > 0.8f;
// 	// 		_mm256_or_si256(row, _mm256_slli_si256(_mm256_castps_si256(_mm256_loadu_ps((float*)&solid)), i));
// 	// 	}
// 	//    _mm256_store_si256(((__m256i*)&bits) + ((k << 8) | j), _mm256_loadu_si256((__m256i_u*)&a));//row);
// 	// }
// }

// void State::Create()
// {
// 	auto timeBefore = std::chrono::high_resolution_clock::now();

// 	std::vector<std::thread> threads;
//     for(int i = 0; i < 256; i++) 
//         threads.push_back(std::thread(&State::fillSliceOfBits, this, i));

//     for(auto &t : threads)
//         if(t.joinable())
//             t.join();

//     // for(int i = 0; i < 256*256*256; i++)
//     // { 
//     //     vec3 pos = vec3((float)(i & 255), (i >> 8) & 255, i>>16);
//     //     bool solid = simplex3D(0.05f * pos) > 0.8f;
//     //     bits.set(i, solid);
//     // }
//     auto timeAfter = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> timeTaken = timeAfter - timeBefore;
//     cout << "BUILDING BITSET time "<< timeTaken.count() << endl;
// }

// bool State::IsSolid(glm::ivec3 pos)
// {
// 	pos = (pos - 128) & 255;
//     return bits.test(pos.x | (pos.y << 8) | (pos.z << 16));
// }
// bool State::IsSolid(glm::vec3 pos)
// {
//     ivec3 ipos = (((ivec3)pos) - 128) & 255;
//     return bits.test(ipos.x | (ipos.y << 8) | (ipos.z << 16));
// }

// hitInfo State::trace(vec3 pos, vec3 dir)
// {
// 	ivec3 ipos = (ivec3)(pos);
// 	ivec3 step = (ivec3)sign(dir);
// 	vec3 deltaDist = abs(1.0f / dir);
// 	//vec3 tMax = (1.0f - glm::fract(pos)) * deltaDist; 
// 	//vec3 tMax = abs((pos - floor(pos) + sign(dir)) * deltaDist);
// 	//vec3 tMax = (sign(dir) * fract(pos)) + ((sign(dir) * 0.5f) + 0.5f) * deltaDist; 

// 	// vec3 tMax = (sign(dir) * (vec3(ipos) - pos) + (sign(dir) * 0.5f) + 0.5f) * deltaDist; 
// 	vec3 tMax = (sign(dir) * (vec3(ipos) - pos) + (sign(dir) * 0.5f) + 0.5f) * deltaDist; 

// 	char mask = -128;

// 	hitInfo HI = hitInfo();
// 	for(int i = 0; i < 4000; i++)
// 	{
// 		if(glm::any(glm::lessThan(ipos, (ivec3)0)) || glm::any(glm::greaterThanEqual(ipos, (ivec3)256)))
// 			break;
// 		if(IsSolid(ipos)) 
// 		{
// 			HI.hit = true;
// 			HI.normal = vec3(mask & 1, 
// 			    	        (mask >> 1) & 1,
// 					         mask >> 2);	
// 			HI.pos = ipos;	
// 			return HI;
// 		}
// 		if(tMax.x < tMax.y) {
// 			if(tMax.x < tMax.z) {
// 				tMax.x += deltaDist.x;
// 				ipos.x += step.x;
// 				mask = 0;
// 			}
// 			else {
// 				tMax.z += deltaDist.z;
// 				ipos.z += step.z;	
// 				mask = 2;
// 			}		
// 		}
// 		else {
// 			if(tMax.y < tMax.z) {
// 				tMax.y += deltaDist.y;
// 				ipos.y += step.y;
// 				mask = 1;
// 			}
// 			else {
// 				tMax.z += deltaDist.z;
// 				ipos.z += step.z;
// 				mask = 2;
// 			}
// 		}
// 	}
// 	HI.hit = false;
	
// 		// mask = sideDist.x < sideDist.y ? (sideDist.y < sideDist.z ? vec3(1, 0, 0) : vec3(0, 1, 0) ) : vec3(0, 0, 1);//glm::step(sideDist.xyz, sideDist.yzx) * glm::(sideDist.xyz, sideDist.zxy);
// 		// sideDist += mask * deltaDist;
// 		// ipos += mask * step;
//     // for(int i = 0; i < 600; i++)
//     // {
//     //     if(IsSolid(pos - 125.f))
//     //         return true;
//     //     pos += dir;
//     // }
//     return HI;
// }

// vec3 State::computeColor(float x, float y)
// {
//     vec3 fo = State::state.character.camera.forward;
//     vec3 up = State::state.character.camera.up;
//     vec3 ri = State::state.character.camera.right;
//     vec3 pos = State::state.character.camera.pos;

//     vec2 NDC = vec2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);

//     vec3 dir = normalize(fo + 
//                          NDC.x * ri +
//                          NDC.y * up);
	
//     hitInfo hit = trace(pos, dir);
//     if(hit.hit) {
// 		float diff = max(dot(hit.normal, sunDir), 0.0f);
// 		return -(vec3)diff;
// 	}

//     return -dir;
// }



// State::State()
// {
// }
// State State::state = State();

// bool State::IsKeyDown(char keycode)
// {
//     return keysPressed.test(keycode);
// }