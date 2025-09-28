
/* 
 * File:   main.c
 * Author: afras
 *
 * Created on September 28, 2025, 12:13 AM
 */

#include <stdio.h>
#include <stdlib.h>
#include "sam.h"

#define MCLK_BASE       0x40000800
#define GCLK_BASE       0x40001C00

#define ADC0_BASE       0x43001C00  

#define PORT_BASE       0x41008000
#define PORTA_BASE      (PORT_BASE + 0)
#define PORTB_BASE      (PORT_BASE + 0x80)

#define SERCOM5_BASE    0x43000400

#define MCLK_APBDMASK   *(volatile uint32_t*)(MCLK_BASE + 0x20) //“ta baseadressen, legg til offset 0x20, behandle det som en peker til en 32-bit volatile int, og les/skriv verdien på den adressen”

#define GCLK_PCHCTRL40  (*(volatile uint32_t*)(GCLK_BASE + 0x80 + (40 * 4)))

#define ADC0_CTRLA      *(volatile uint16_t*)(ADC0_BASE + 0x00)
#define ADC0_INPUTCTRL  *(volatile uint16_t*)(ADC0_BASE + 0x04)
#define ADC0_INTFLAG    *(volatile uint32_t*)(ADC0_BASE + 0x2E)
#define ADC0_RESULT     *(volatile uint16_t*)(ADC0_BASE + 0x40)
#define ADC0_SWTRIG     *(volatile uint32_t*)(ADC0_BASE + 0x14)

#define PORTA_PINCFG3   *(volatile uint8_t*)(PORTA_BASE + 0x40 + 3)
#define PORTA_PMUX1     *(volatile uint8_t*)(PORTA_BASE + 0x30 + 1)

#define 

int main(int argc, char** argv) {
    // Enable ADC0 klokke
    MCLK_APBDMASK |= (1 << 7);
    
    GCLK_PCHCTRL40 = (1 << 6) | (0 << 0);
    
    // Config PA03 pin
    PORTA_PINCFG3 |= ((1 << 0) | (1 << 1));
    PORTA_PMUX1 = (PORTA_PMUX1 & 0x0F) | (0x1 << 4);
    
    // ADC setup
    ADC0_INPUTCTRL = (ADC0_INPUTCTRL & ~0x1F) | 0x01;
    ADC0_CTRLA |= (1 << 1);
    
    // read
    while (1) {
        ADC0_SWTRIG |= (1 << 1);
        while (!(ADC0_INTFLAG & (1 << 0)));
        volatile uint16_t result = ADC0_RESULT;
    }
    

    return (EXIT_SUCCESS);
}

