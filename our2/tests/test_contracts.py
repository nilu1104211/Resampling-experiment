"""Failure-prone contracts: discrete budgets, capacities, independent inputs, stages."""
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT/'PUFM-main'))
from our2.config import DEFAULT,validate_config
from models.gdba import bounded_soft_budget,integer_budget,GDBA
from models.geometry import geometry
from models.dual import DualPUFM
from models.adaptive_ops import midpoint,knn


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_budget_extremes_and_infeasibility(self):
        torch.manual_seed(2)
        for k,n in [(1,37),(8,1024),(64,8192),(7,1003)]:
            lower=torch.full((2,k),int(.5*n/k))
            upper=torch.full((2,k),int(2*n/k))
            for scale in (0,1,10000):
                logits=torch.randn(2,k)*scale
                soft=bounded_soft_budget(logits,n,lower,upper)
                hard=integer_budget(soft,n,lower,upper)
                self.assertTrue((hard.sum(-1)==n).all())
                self.assertTrue(((hard>=lower)&(hard<=upper)).all())
                torch.testing.assert_close(soft.sum(-1),torch.full((2,),float(n)),atol=.01,rtol=0)
        with self.assertRaises(ValueError): bounded_soft_budget(torch.zeros(1,2),10,torch.zeros(1,2),torch.ones(1,2))

    def test_soft_budget_gradient(self):
        logits=torch.tensor([[.1,.4,-.2]],requires_grad=True)
        soft=bounded_soft_budget(logits,101,torch.tensor([[10,10,10]]),torch.tensor([[80,80,80]]))
        (soft*torch.tensor([[1.,2.,3.]])).sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_region_accounting_and_capacity(self):
        torch.manual_seed(5)
        points=torch.randn(1,256,3)
        geo=geometry(points)
        model=GDBA().eval()
        seeds=midpoint(points)
        result,state=model.allocate(seeds,geo,model(geo))
        self.assertEqual(len(torch.unique(result[0],dim=0)),1024)
        labels=knn(geo['anchors'],result,1).squeeze(-1)
        actual=torch.nn.functional.one_hot(labels,8).sum(1)
        torch.testing.assert_close(actual,state['hard'])
        with self.assertRaises(ValueError):
            points=torch.zeros(1,256,3)
            geo=geometry(points)
            model.allocate(midpoint(points),geo,model(geo))

    def test_m2_independence_and_frozen_bn(self):
        geo=geometry(torch.randn(1,256,3))
        m2=GDBA().eval()
        fake={'r':torch.zeros(1,256),'h':torch.zeros(1,8)}
        torch.testing.assert_close(m2(geo,None,False),m2(geo,fake,False),atol=0,rtol=0)
        config={**DEFAULT,'enable_m1':True,'enable_m2':True,'stage':'joint_condition'}
        model=DualPUFM(config).train()
        self.assertFalse(model.flow.training)
        self.assertFalse(model.m1.training)
        self.assertTrue(model.m2.training)
        self.assertTrue(all(not p.requires_grad for p in model.m1.parameters()))

    def test_config_contract(self):
        validate_config(DEFAULT)
        with self.assertRaises(ValueError): validate_config({**DEFAULT,'enable_m2':True})
        with self.assertRaises(ValueError): validate_config({**DEFAULT,'minimum':2})


if __name__=='__main__': unittest.main()
