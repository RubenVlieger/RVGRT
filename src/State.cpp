#include "State.hpp"
#define NOMINMAX
#include <thread>
#include <vector>
#include <iostream>
#include <StateRender.cuh>
#include <CArray.cuh>
#include "Timer.hpp"

#include "CSDF.cuh"
#include "CuTex.cuh"

State::State() 
{
	render = new StateRender();
}

State State::state = State();

bool State::IsKeyDown(char keycode) 
{
    return keysPressed.test(keycode);
}

void State::Create() 
{
    Timer t1("allocating");
    
    render->distBuffer = CArray();
    render->distBuffer.Allocate((dispWIDTH/2) * (dispHEIGHT/2) * 2);

    render->cArray = CArray();
    render->cArray.Allocate(BYTESIZE);

    render->bitsArray = new uint32_t[BYTESIZE / 4];

    render->csdf = CSDF();
    render->csdf.Allocate();

    render->shadowTex = CuTex(dispWIDTH/2, dispHEIGHT/2, cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat),cudaAddressModeWrap ,cudaFilterModeLinear);

    t1.s();

    Timer t2("BUILDING FINE ARRAY");
    render->cArray.fill();
    render->cArray.readback(render->bitsArray);
    t2.s();

    Timer t3("BUILDING CSDF");
    render->csdf.Generate(render->cArray);
    t3.s();
}
