struct PS_Input {
    float4 position : SV_POSITION;
    float3 fragColor : TEXCOORD0;
};

struct PS_Output {
    float4 outColor : SV_TARGET0;
};


PS_Output main(PS_Input input) {    
    float4 outFragColor = float4(input.fragColor, 1.0);
    
    PS_Output output;
    output.outColor = float4(outFragColor);
    
    return output;
}
