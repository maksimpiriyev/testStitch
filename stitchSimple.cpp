
#include <arrayfire.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
 
using namespace af;
 

dim4 operator*(const dim4 &d, const int c)
{
    dim4 rtn;
    rtn[0] = d[0]*c;
    rtn[1] = d[1]*c;
    rtn[2] = d[2]*c;
    rtn[3] = d[3]*c;
    return rtn;
}
dim_t max(dim_t a,dim_t b){ return a > b ? a : b;}
dim4  max(dim4 a, dim4 b){
    dim4 rtn;
    rtn[0] = max(a[0],b[0]);
    rtn[1] = max(a[1],b[1]);
    rtn[2] = max(a[2],b[2]);
    rtn[3] = max(a[3],b[3]);
    return rtn;
}

void stitchOrb(){
    auto img1 = loadimage ("newspaper1.jpg", true);
    auto img2 = loadimage ("newspaper2.jpg", true);
    
    auto img1g = rgb2gray (img1);
    auto img2g = rgb2gray (img2);
    auto img4 = img1*0.5 + img2*0.5;
    auto img5 = rotate(img1, M_PI/2,false,AF_INTERP_BILINEAR);
    
    auto dims = max(img1.dims(),img5.dims());

    
    features feat1,feat2;
    array desc1,desc2;
//    af_orb(<#af_features *feat#>, <#af_array *desc#>, <#const af_array in#>, <#const float fast_thr#>, <#const unsigned int max_feat#>, <#const float scl_fctr#>, <#const unsigned int levels#>, <#const bool blur_img#>)
    orb(feat1, desc1, img1g, 20, 7000, 1.2, 8, true);
    orb(feat2, desc2, img2g, 20, 7000, 1.2, 8, true);
    
    array idx,dist;
    hammingMatcher(idx, dist, desc1, desc2);
    
  
    array near_matches = where(dist < 100);
    array near_idx = idx(near_matches);
  
    array a_feat_x = feat1.getX()(near_matches);
    array a_feat_y = feat1.getY()(near_matches);
    
    array b_feat_x = feat2.getX()(near_idx);
    array b_feat_y = feat2.getY()(near_idx);
    
    printf("num feat: %d\n",feat1.getNumFeatures());
    
    array H;
    int inliers = 0;
//  af_print(a_feat_x);
//  af_print(b_feat_x);
    
    homography(H, inliers, a_feat_x, a_feat_y, b_feat_x, b_feat_y,
               AF_HOMOGRAPHY_RANSAC, 3.0f, 1000, f32);
    af_print(H);

    printf("inliers: %d\n",inliers);
    
    array img7 = transform(img2g, H);
    
    dims = img7.dims();
    printf("%d\n",dims.dims[0]);
    printf("%d\n",dims.dims[1]);
    
    
    
    saveimage ("image3.jpg", img7);
        
    float* h_x = feat2.getX().host<float>();
    float* h_y = feat2.getY().host<float>();
    
    
    auto img_color = img2;
    img_color /= 255.f;
    
       // Draw draw_len x draw_len crosshairs where the corners are
       const int draw_len = 3;
       //for (size_t f = 0; f < b_feat_x.dims()[0]; f++) {
      for (size_t f = 0; f < feat2.getNumFeatures(); f++) {
           int x                                            = h_x[f];
           int y                                            = h_y[f];
           img_color(y, seq(x - draw_len, x + draw_len), 0) = 0.f;
           img_color(y, seq(x - draw_len, x + draw_len), 1) = 1.f;
           img_color(y, seq(x - draw_len, x + draw_len), 2) = 0.f;
    
           // Draw vertical line of (draw_len * 2 + 1) pixels centered on  the
           // corner Set only the first channel to 1 (green lines)
           img_color(seq(y - draw_len, y + draw_len), x, 0) = 0.f;
           img_color(seq(y - draw_len, y + draw_len), x, 1) = 1.f;
           img_color(seq(y - draw_len, y + draw_len), x, 2) = 0.f;
       }
    
       freeHost(h_x);
       freeHost(h_y);
    
       printf("Features found: %zu\n", feat1.getNumFeatures());
    
       if (true) {
           af::Window wnd("FAST Feature Detector");

           // Previews color image with green crosshairs
           while (!wnd.close()) wnd.image(img_color);
       } else {
           af_print(feat1.getX());
           af_print(feat1.getY());
       }
}

int main(int argc, char** argv) {
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        setDevice(device);
        info();
 
        stitchOrb();
                
    } catch (af::exception& e) { fprintf(stderr, "%s\n", e.what()); }
 
    return 0;
}
