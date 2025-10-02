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
#define MCLK_APBDMASK   (volatile uint32_t)(MCLK_BASE + 0x20) // ta baseadressen, legg til offset 0x20, behandle det som en peker til en 32-bit volatile int, og les/skriv verdien på den adressen?
#define GCLK_PCHCTRL40  ((volatile uint32_t)(GCLK_BASE + 0x80 + (40 * 4)))
#define GCLK_PCHCTRL35 ((volatile uint32_t)(GCLK_BASE + 0x80 + (35 * 4)))
#define ADC0_CTRLA      (volatile uint16_t)(ADC0_BASE + 0x00)
#define ADC0_INPUTCTRL  (volatile uint16_t)(ADC0_BASE + 0x04)
#define ADC0_INTFLAG    (volatile uint8_t)(ADC0_BASE + 0x2E)
#define ADC0_RESULT     (volatile uint16_t)(ADC0_BASE + 0x40)
#define ADC0_SWTRIG     (volatile uint8_t)(ADC0_BASE + 0x14)
#define ADC0_SYNCBUSY   (volatile uint32_t)(ADC0_BASE + 0x30)
#define PORTA_PINCFG3   (volatile uint8_t)(PORTA_BASE + 0x40 + 3)
#define PORTA_PMUX1     (volatile uint8_t)(PORTA_BASE + 0x30 + 1)
#define PORTB_PINCFG16  (volatile uint8_t)(PORTB_BASE + 0x40 + 16)
#define PORTB_PINCFG17  (volatile uint8_t)(PORTB_BASE + 0x40 + 17)
#define PORTB_PMUX8     (volatile uint8_t)(PORTB_BASE + 0x30 + 8)
#define SERCOM5_CTRLA       (volatile uint32_t)(SERCOM5_BASE + 0x00)
#define SERCOM5_CTRLB       (volatile uint32_t)(SERCOM5_BASE + 0x04)
#define SERCOM5_BAUD        (volatile uint16_t)(SERCOM5_BASE + 0x0C)
#define SERCOM5_DATA        (volatile uint32_t)(SERCOM5_BASE + 0x28)
#define SERCOM5_INTFLAG     (volatile uint8_t)(SERCOM5_BASE + 0x18)
#define SERCOM5_SYNCBUSY    (volatile uint32_t)(SERCOM5_BASE + 0x1C)
void uart_send_char(char c) {
    while (!(SERCOM5_INTFLAG & (1 << 0)));            
    SERCOM5_DATA = c;
}
void uart_send_number(uint16_t num) {
    char buffer[10];
    sprintf(buffer, "%u\r\n", num);
    for (int i = 0; buffer[i]; i++) {
        uart_send_char(buffer[i]);
    }
}
int main(int argc, char** argv) {
    // Enable ADC0 og sercom5 klokke
    MCLK_APBDMASK |= (1 << 7) | (1 << 1);
    GCLK_PCHCTRL40 = (1 << 6) | (0 << 0);
    GCLK_PCHCTRL35 = (1 << 6) | (0x0 << 0);  

    // Config PA03 pin
    PORTA_PINCFG3 |= ((1 << 0) | (1 << 1));
    PORTA_PMUX1 = (PORTA_PMUX1 & 0x0F) | (0x1 << 4);

    // ADC setup
    ADC0_CTRLA |= (1 << 0);
    while (ADC0_SYNCBUSY & (1 << 0));

    ADC0_INPUTCTRL = (ADC0_INPUTCTRL & ~0x1F) | 0x01;
    while (ADC0_SYNCBUSY & (1 << 2));

    ADC0_CTRLA |= (1 << 1);
    while (ADC0_SYNCBUSY & (1 << 1));

    for(int i = 0; i < 100000; i++);
    // Configure PB16/PB17 for SERCOM   
    PORTB_PINCFG16  |= (1 << 0);
    PORTB_PINCFG17  |= (1 << 0);
    PORTB_PMUX8      = (0x2 << 4) | (0x2 << 0);

    // Uart
    SERCOM5_CTRLA = (1 << 0);
    while (SERCOM5_SYNCBUSY & (1 << 0));  

    SERCOM5_CTRLA    = (0x1 << 2) | (0 << 16) | (1 << 20) | (1 << 30);
    while (SERCOM5_SYNCBUSY & (1 << 1));

    SERCOM5_BAUD     = 0xF62B;

    SERCOM5_CTRLB = (1 << 16) | (1 << 17) | (0x0 << 0);
    while (SERCOM5_SYNCBUSY & (1 << 2));   

    SERCOM5_CTRLA   |= (1 << 1);    
    while (SERCOM5_SYNCBUSY & (1 << 1));

    // read
    while (1) {
        ADC0_SWTRIG |= (1 << 1);
        while (!(ADC0_INTFLAG & (1 << 0)));
        //uint16_t result = ADC0_RESULT;
        ADC0_INTFLAG = (1 << 0);   

        //uart_send_number(result);
        uart_send_char('A');
        uart_send_char('B');
        uart_send_char('C');
        uart_send_char('D');
        uart_send_char('E');

        for(int i = 0; i < 100000; i++);
    }

    return (0);
}
