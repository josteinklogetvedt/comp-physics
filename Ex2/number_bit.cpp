#include <iostream>
#include <climits>

void print_int(int num){
    size_t nBytes = sizeof(num);
    for (int i=nBytes*CHAR_BIT-1; i>=0; i--){
        bool bit = num & (1 << i);
        std::cout << bit;
        if (i % 8 == 0)
            printf(" ");
    }
    std::cout << std::endl;
}

void print_float(float * fl){
    size_t nBytes = sizeof(fl);
    int * c = (int*)fl;
    for (int i=nBytes*CHAR_BIT-1; i>=0; i--){
        bool bit = *c & (1 << i);
        std::cout << bit;
        if (i == 31 || i == 23)
            printf(" ");
    }
    std::cout << std::endl;
}

void print_double(double * fl){

    long long int * c = (long long int*)fl;
    for (int i=63; i>=0; i--){
        bool bit = *c & ((long long int)1 << i);
        std::cout << bit;
        if (i == 63 || i == 52)
            printf(" ");
    }
    std::cout << std::endl;
}

int main(){
    print_int(5);

    float f1 = -0.4765625;
    print_float(&f1);

    double f2 = -0.4765625;
    print_double(&f2);

    printf("Finding max precision:\n");

    double  pres = 1.0;
    
    while (1.0 + pres > 1.0) {
        pres = pres * 0.5;
        //printf("pres = %.18lf  ",pres);
        //printf("1.0 + pres = %.18lf\n",pres+1.0);
    }
    pres = pres*2.0;
    printf("pres=%.18g\n",pres);
    printf("1.0 + pres=%.18g\n",pres+1.0);
    
    return 0;    
}